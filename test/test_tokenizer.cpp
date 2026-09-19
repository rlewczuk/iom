#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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

TEST_CASE("Tokenizer encodes empty whitespace and ranked merge cases exactly") {
    TempDir directory("encode-basic");
    write_tokenizer(directory.path());
    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    CHECK(owner->encode("", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{});
    CHECK(owner->encode("", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1});
    CHECK(owner->encode(" ", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 259});
    CHECK(owner->encode("  ", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 259, 259});
    CHECK(owner->encode("hello", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{268});
    CHECK(owner->encode("hello", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 268});
    CHECK(owner->encode("hello hello", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{268, 268});
    CHECK(owner->encode("hello hello", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 268, 268});
    CHECK(owner->encode("\t", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 12});
    CHECK(owner->encode("\n", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 13});
}

TEST_CASE("Tokenizer encodes Unicode scalars bytes and ordinary role markers") {
    TempDir directory("encode-unicode");
    write_tokenizer(directory.path());
    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    CHECK(owner->encode("héllø 世界", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 260, 277, 265, 278, 259, 279, 280});
    CHECK(owner->encode("héllø 世界", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{
                  1, 259, 260, 277, 265, 278, 259, 279, 280});
    std::string u_d7ff(3, '\0');
    u_d7ff[0] = static_cast<char>(0xED);
    u_d7ff[1] = static_cast<char>(0x9F);
    u_d7ff[2] = static_cast<char>(0xBF);
    CHECK(owner->encode(u_d7ff, iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 240, 162, 194});
    CHECK(owner->encode(u_d7ff, iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 259, 240, 162, 194});
    CHECK(owner->encode("🙂", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 243, 162, 156, 133});
    CHECK(owner->encode("<|user|>", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{
                  259, 63, 127, 120, 118, 261, 270, 127, 65});
    CHECK(owner->encode("!", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{259, 283});
}

TEST_CASE("Tokenizer segments added tokens and applies only the single BOS template") {
    TempDir directory("encode-specials");
    write_tokenizer(directory.path());
    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    CHECK(owner->encode("<unk>hello<s></s>", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{0, 268, 1, 2});
    CHECK(owner->encode("<unk>hello<s></s>", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 0, 268, 1, 2});
    CHECK(owner->encode("<s>", iom::EncodeOptions{false}) ==
          std::vector<std::uint32_t>{1});
    CHECK(owner->encode("<s>", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 1});
    CHECK(owner->encode("</s>", iom::EncodeOptions{true}) ==
          std::vector<std::uint32_t>{1, 2});
}

TEST_CASE("Tokenizer rejects invalid UTF-8 before publishing an encode result") {
    TempDir directory("encode-invalid-utf8");
    write_tokenizer(directory.path());
    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    std::string invalid(4, '\0');
    invalid[0] = static_cast<char>(0xF0);
    invalid[1] = static_cast<char>(0x28);
    invalid[2] = static_cast<char>(0x8C);
    invalid[3] = static_cast<char>(0x28);
    for (const bool add_special_tokens : {false, true}) {
        std::vector<std::uint32_t> published{777};
        bool rejected = false;
        std::string message;
        try {
            const auto actual =
                    owner->encode(invalid, iom::EncodeOptions{add_special_tokens});
            published = actual;
        } catch (const std::invalid_argument& error) {
            rejected = true;
            message = error.what();
        }
        CHECK(rejected);
        CHECK(published == std::vector<std::uint32_t>{777});
        CHECK(message.find("Tokenizer::encode") != std::string::npos);
        CHECK(message.find("invalid UTF-8") != std::string::npos);
        CHECK(message.find("byte 1") != std::string::npos);
        CHECK(message.find("expected") != std::string::npos);
        CHECK(message.find("actual") != std::string::npos);
    }
}

// Decode tests
TEST_CASE("Tokenizer decodes sentence pieces, byte fallback, and decoder boundaries") {
    TempDir directory("decode-sequence");
    write_tokenizer(directory.path());

    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    const std::array<std::uint32_t, 0> empty{};
    CHECK(owner->decode(empty, {}) == "");

    const std::array<std::uint32_t, 3> sentence = {268, 276, 282};
    CHECK(owner->decode(sentence, {}) == "hello world 世界");

    const std::array<std::uint32_t, 5> fused = {260, 261, 262, 262, 263};
    CHECK(owner->decode(fused, {}) == "hello");

    const std::array<std::uint32_t, 4> spaces = {259, 259, 267, 259};
    CHECK(owner->decode(spaces, {}) == " hello ");

    const std::array<std::uint32_t, 1> unicode = {281};
    CHECK(owner->decode(unicode, {}) == "héllø");

    const std::array<std::uint32_t, 4> bytes = {243, 43, 143, 43};
    std::string expected_bytes;
    expected_bytes.push_back(static_cast<char>(0xF0));
    expected_bytes.push_back('(');
    expected_bytes.push_back(static_cast<char>(0x8C));
    expected_bytes.push_back('(');
    CHECK(owner->decode(bytes, {}) == expected_bytes);
}

TEST_CASE("Tokenizer decodes only declared special IDs and ignores decoys") {
    TempDir directory("decode-specials");
    json document = make_tokenizer_document();
    document["model"]["vocab"].erase("<filler-285>");
    document["model"]["vocab"]["<|user|>"] = 285;
    document["model"]["vocab"].erase("<filler-286>");
    document["model"]["vocab"]["literal<0x00>"] = 286;
    write_tokenizer(directory.path(), document);
    write_file(directory.path(), "tokenizer.model", "not a model");
    write_file(directory.path(), "tokenizer_config.json", "{not json");
    write_file(directory.path(), "generation_config.json", "[]");

    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    const std::array<std::uint32_t, 6> ids = {0, 1, 2, 285, 286, 3};
    std::string retained = "<unk><s></s><|user|>literal<0x00>";
    retained.push_back('\0');
    CHECK(owner->decode(ids, {}) == retained);

    std::string skipped = "<|user|>literal<0x00>";
    skipped.push_back('\0');
    CHECK(owner->decode(ids, iom::DecodeOptions{true}) == skipped);
}

TEST_CASE("Tokenizer validates every decode ID before publishing output") {
    TempDir directory("decode-invalid");
    write_tokenizer(directory.path());

    const auto owner = iom::load_tokenizer(directory.path());
    REQUIRE(owner);

    std::string published = "sentinel";
    const auto expect_invalid = [&](std::span<const std::uint32_t> ids,
                                    std::size_t index, std::uint32_t id,
                                    iom::DecodeOptions options) {
        bool rejected = false;
        std::string message;
        try {
            published = owner->decode(ids, options);
        } catch (const std::invalid_argument& error) {
            rejected = true;
            message = error.what();
        }
        CHECK(rejected);
        CHECK(published == "sentinel");
        CHECK(message.find("index " + std::to_string(index)) !=
              std::string::npos);
        CHECK(message.find(std::to_string(id)) != std::string::npos);
        CHECK(message.find("0..31999") != std::string::npos);
    };

    const std::array<std::uint32_t, 2> first = {32'000, 267};
    expect_invalid(first, 0, 32'000, iom::DecodeOptions{true});

    const std::array<std::uint32_t, 3> middle = {
            267, std::numeric_limits<std::uint32_t>::max(), 267};
    expect_invalid(middle, 1, std::numeric_limits<std::uint32_t>::max(), {});

    const std::array<std::uint32_t, 2> last = {267, 32'000};
    expect_invalid(last, 1, 32'000, iom::DecodeOptions{true});

    const std::array<std::uint32_t, 1> valid = {267};
    CHECK(owner->decode(valid, {}) == "hello");
}
}  // namespace
