#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace iom_tokenizer_test {

class TempDir {
public:
    explicit TempDir(std::string tag)
        : path_(std::filesystem::temp_directory_path() /
                ("iom-tokenizer-" + std::move(tag))) {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (!std::filesystem::create_directories(path_)) {
            throw std::runtime_error(
                    "cannot create tokenizer test directory: " + path_.string());
        }
    }

    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

inline std::filesystem::path write_file(const std::filesystem::path& directory,
                                         std::string_view filename,
                                         std::string_view contents) {
    const std::filesystem::path path = directory / filename;
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot write tokenizer test file: " +
                                 path.string());
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.close();
    if (!stream) {
        throw std::runtime_error("cannot write tokenizer test file: " +
                                 path.string());
    }
    return path;
}

inline std::string byte_piece(std::uint32_t byte) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string result = "<0x00>";
    result[3] = kHex[(byte >> 4) & 0x0F];
    result[4] = kHex[byte & 0x0F];
    return result;
}

// Builds the complete frozen descriptor independently of any model download or
// sibling task output. Its small merge set exercises each kind of BPE
// cross-reference while the vocabulary still has the production cardinality.
inline nlohmann::json make_tokenizer_document() {
    using nlohmann::json;

    json document = json::object();
    document["version"] = "1.0";
    document["truncation"] = nullptr;
    document["padding"] = nullptr;

    document["added_tokens"] = json::array({
            json{{"id", 0},
                 {"content", "<unk>"},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}},
            json{{"id", 1},
                 {"content", "<s>"},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}},
            json{{"id", 2},
                 {"content", "</s>"},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}},
    });

    document["normalizer"] = json{
            {"type", "Sequence"},
            {"normalizers",
             json::array({json{{"type", "Prepend"}, {"prepend", "▁"}},
                          json{{"type", "Replace"},
                               {"pattern", json{{"String", " "}}},
                               {"content", "▁"}}})}};
    document["pre_tokenizer"] = nullptr;

    document["post_processor"] = json{
            {"type", "TemplateProcessing"},
            {"single",
             json::array({json{{"SpecialToken",
                                json{{"id", "<s>"}, {"type_id", 0}}}},
                          json{{"Sequence", json{{"id", "A"}, {"type_id", 0}}}}})},
            {"pair",
             json::array({json{{"SpecialToken",
                                json{{"id", "<s>"}, {"type_id", 0}}}},
                          json{{"Sequence", json{{"id", "A"}, {"type_id", 0}}}},
                          json{{"SpecialToken",
                                json{{"id", "<s>"}, {"type_id", 1}}}},
                          json{{"Sequence", json{{"id", "B"}, {"type_id", 1}}}}})},
            {"special_tokens",
             json{{"<s>",
                   json{{"id", "<s>"}, {"ids", json::array({1})},
                        {"tokens", json::array({"<s>"})}}}}}};

    document["decoder"] = json{
            {"type", "Sequence"},
            {"decoders",
             json::array({
                     json{{"type", "Replace"},
                          {"pattern", json{{"String", "▁"}}},
                          {"content", " "}},
                     json{{"type", "ByteFallback"}},
                     json{{"type", "Fuse"}},
                     json{{"type", "Strip"},
                          {"content", " "},
                          {"start", 1},
                          {"stop", 0}},
             })}};

    json vocab = json::object();
    vocab["<unk>"] = 0;
    vocab["<s>"] = 1;
    vocab["</s>"] = 2;
    for (std::uint32_t byte = 0; byte <= 0xFF; ++byte) {
        vocab[byte_piece(byte)] = byte + 3;
    }

    const std::vector<std::string> recipe = {
            "▁",       "h",       "e",       "l",       "o",       "he",
            "ll",      "llo",     "hello",   "▁hello",  "w",       "r",
            "d",        "wo",      "wor",     "worl",    "world",   "▁world",
            "é",        "ø",        "世",        "界",        "▁héllø", "▁世界",
            "!",        "?"};
    std::uint32_t next_id = 259;
    for (const std::string& token : recipe) {
        vocab[token] = next_id++;
    }
    while (next_id < 32'000) {
        const std::string filler = "<filler-" + std::to_string(next_id) + ">";
        vocab[filler] = next_id++;
    }

    json model = json::object();
    model["type"] = "BPE";
    model["dropout"] = nullptr;
    model["unk_token"] = "<unk>";
    model["continuing_subword_prefix"] = nullptr;
    model["end_of_word_suffix"] = nullptr;
    model["fuse_unk"] = true;
    model["byte_fallback"] = true;
    model["vocab"] = std::move(vocab);
    model["merges"] = json::array({"h e", "l l", "ll o", "he llo",
                                    "▁ hello", "w o", "wo r", "wor l",
                                    "worl d", "▁ world"});
    document["model"] = std::move(model);
    return document;
}

inline nlohmann::json valid_tokenizer_document() {
    return make_tokenizer_document();
}

inline std::filesystem::path write_tokenizer(
        const std::filesystem::path& directory,
        const nlohmann::json& document = make_tokenizer_document()) {
    return write_file(directory, "tokenizer.json", document.dump());
}

}  // namespace iom_tokenizer_test
