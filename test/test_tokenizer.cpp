#include <doctest/doctest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "iom/tokenizer.hpp"
#include "tokenizer_fixture.hpp"

namespace {

using iom_tokenizer_test::TempDir;
using iom_tokenizer_test::make_tokenizer_document;
using iom_tokenizer_test::write_file;
using iom_tokenizer_test::write_tokenizer;
using nlohmann::json;

void expect_invalid(const json& document, const std::string& tag,
                    std::string_view field) {
    TempDir directory(tag);
    write_tokenizer(directory.path(), document);

    bool rejected = false;
    std::string message;
    try {
        const auto owner = iom::load_tokenizer(directory.path());
        (void)owner;
    } catch (const std::invalid_argument& error) {
        rejected = true;
        message = error.what();
    }
    CHECK(rejected);
    REQUIRE_FALSE(message.empty());
    CHECK(message.find(directory.path().string()) != std::string::npos);
    CHECK(message.find(field) != std::string::npos);
}

TEST_CASE("Tokenizer loads the complete synthetic artifact and exposes frozen metadata") {
    TempDir directory("valid");
    write_tokenizer(directory.path());

    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);
    CHECK(owner->bos_id() == 1);
    CHECK(owner->eos_id() == 2);
    CHECK(owner->unk_id() == 0);
    CHECK(owner->tokenizer_pad_id() == 2);
}

TEST_CASE("Tokenizer ignores corrupt non-authoritative tokenizer sidecars") {
    TempDir directory("decoys");
    write_tokenizer(directory.path());
    write_file(directory.path(), "tokenizer.model", "not a model");
    write_file(directory.path(), "tokenizer_config.json", "{not json");
    write_file(directory.path(), "generation_config.json", "[]");

    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);
    CHECK(owner->bos_id() == 1);
    CHECK(owner->eos_id() == 2);
    CHECK(owner->unk_id() == 0);
    CHECK(owner->tokenizer_pad_id() == 2);
}

TEST_CASE("Tokenizer factory owns a final non-copyable non-movable owner") {
    static_assert(std::is_same_v<decltype(iom::load_tokenizer(
                                         std::declval<std::filesystem::path>())),
                                 std::unique_ptr<iom::Tokenizer>>);
    static_assert(!std::is_copy_constructible_v<iom::Tokenizer>);
    static_assert(!std::is_copy_assignable_v<iom::Tokenizer>);
    static_assert(!std::is_move_constructible_v<iom::Tokenizer>);
    static_assert(!std::is_move_assignable_v<iom::Tokenizer>);

    CHECK(std::is_final_v<iom::Tokenizer>);
}

TEST_CASE("Tokenizer reports a missing artifact as contextual I/O failure") {
    TempDir directory("missing");
    bool rejected_as_argument = false;
    std::string message;
    try {
        const auto owner = iom::load_tokenizer(directory.path());
        (void)owner;
    } catch (const std::invalid_argument&) {
        rejected_as_argument = true;
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    CHECK_FALSE(rejected_as_argument);
    REQUIRE_FALSE(message.empty());
    CHECK(message.find(directory.path().string()) != std::string::npos);
    CHECK(message.find("tokenizer.json") != std::string::npos);
    CHECK(message.find("open") != std::string::npos);
}

TEST_CASE("Tokenizer rejects malformed JSON before constructing an owner") {
    TempDir directory("malformed-json");
    write_file(directory.path(), "tokenizer.json", "{not valid JSON");

    std::unique_ptr<iom::Tokenizer> owner;
    bool rejected = false;
    std::string message;
    try {
        owner = iom::load_tokenizer(directory.path());
    } catch (const std::invalid_argument& error) {
        rejected = true;
        message = error.what();
    }
    CHECK(rejected);
    CHECK_FALSE(owner);
    CHECK(message.find(directory.path().string()) != std::string::npos);
    CHECK(message.find("<document>") != std::string::npos);
}

TEST_CASE("Tokenizer rejects unsupported top-level and null schema values") {
    json unknown = make_tokenizer_document();
    unknown["unexpected"] = true;
    expect_invalid(unknown, "unknown-top-level", "unexpected");

    json truncation = make_tokenizer_document();
    truncation["truncation"] = json::object();
    expect_invalid(truncation, "truncation", "truncation");

    json padding = make_tokenizer_document();
    padding["padding"] = false;
    expect_invalid(padding, "padding", "padding");

    json pre_tokenizer = make_tokenizer_document();
    pre_tokenizer["pre_tokenizer"] = json::object();
    expect_invalid(pre_tokenizer, "pre-tokenizer", "pre_tokenizer");
}

TEST_CASE("Tokenizer validates added token identity and flags") {
    json count = make_tokenizer_document();
    count["added_tokens"].erase(2);
    expect_invalid(count, "added-count", "added_tokens");

    json id = make_tokenizer_document();
    id["added_tokens"][1]["id"] = 7;
    expect_invalid(id, "added-id", "added_tokens[1].id");

    json content = make_tokenizer_document();
    content["added_tokens"][0]["content"] = "<bad>";
    expect_invalid(content, "added-content", "added_tokens[0].content");

    json flag = make_tokenizer_document();
    flag["added_tokens"][2]["special"] = false;
    expect_invalid(flag, "added-flag", "added_tokens[2].special");

    json extra = make_tokenizer_document();
    extra["added_tokens"][0]["extra"] = 1;
    expect_invalid(extra, "added-extra", "added_tokens[0].extra");
}

TEST_CASE("Tokenizer validates normalizer and processor structure") {
    json normalizer = make_tokenizer_document();
    std::swap(normalizer["normalizer"]["normalizers"][0],
              normalizer["normalizer"]["normalizers"][1]);
    expect_invalid(normalizer, "normalizer-order", "normalizer.normalizers[0]");

    json pattern = make_tokenizer_document();
    pattern["normalizer"]["normalizers"][1]["pattern"]["String"] = "_";
    expect_invalid(pattern, "normalizer-pattern", "normalizer.normalizers[1].pattern");

    json single = make_tokenizer_document();
    single["post_processor"]["single"][0]["SpecialToken"]["type_id"] = 1;
    expect_invalid(single, "single-template", "post_processor.single");

    json pair = make_tokenizer_document();
    pair["post_processor"]["pair"].erase(3);
    expect_invalid(pair, "pair-template", "post_processor.pair");

    json special = make_tokenizer_document();
    special["post_processor"]["special_tokens"]["<s>"]["ids"][0] = 2;
    expect_invalid(special, "special-table", "special_tokens.<s>.ids");
}

TEST_CASE("Tokenizer validates decoder order and BPE options") {
    json decoder = make_tokenizer_document();
    std::swap(decoder["decoder"]["decoders"][0],
              decoder["decoder"]["decoders"][1]);
    expect_invalid(decoder, "decoder-order", "decoder.decoders[0]");

    json strip = make_tokenizer_document();
    strip["decoder"]["decoders"][3]["start"] = 2;
    expect_invalid(strip, "decoder-strip", "decoder.decoders[3].start");

    json model = make_tokenizer_document();
    model["model"]["byte_fallback"] = false;
    expect_invalid(model, "model-option", "model.byte_fallback");

    json unknown = make_tokenizer_document();
    unknown["model"]["merges"] = json::array({"missing piece"});
    expect_invalid(unknown, "merge-count-validity", "model.merges[0]");
}

TEST_CASE("Tokenizer validates vocabulary IDs special pieces and merge cross references") {
    json duplicate_id = make_tokenizer_document();
    duplicate_id["model"]["vocab"]["<s>"] = 0;
    expect_invalid(duplicate_id, "vocab-duplicate-id", "model.vocab.");

    json gap = make_tokenizer_document();
    gap["model"]["vocab"]["<s>"] = 31999;
    expect_invalid(gap, "vocab-gap", "model.vocab");

    json bad_byte = make_tokenizer_document();
    bad_byte["model"]["vocab"]["<0x00>"] = "wrong";
    expect_invalid(bad_byte, "vocab-byte", "model.vocab.<0x00>");

    json missing_result = make_tokenizer_document();
    missing_result["model"]["merges"][0] = "h ?";
    expect_invalid(missing_result, "merge-result", "model.merges[0]");

    json malformed = make_tokenizer_document();
    malformed["model"]["merges"][0] = "h  e";
    expect_invalid(malformed, "merge-malformed", "model.merges[0]");

    json duplicate_merge = make_tokenizer_document();
    duplicate_merge["model"]["merges"].push_back("h e");
    expect_invalid(duplicate_merge, "merge-duplicate", "model.merges[10]");
}

}  // namespace
