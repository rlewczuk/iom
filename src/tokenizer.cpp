#include "iom/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace iom {
namespace {

using Json = nlohmann::json;

constexpr std::uint32_t kUnkId = 0;
constexpr std::uint32_t kBosId = 1;
constexpr std::uint32_t kEosId = 2;
constexpr std::uint32_t kVocabularySize = 32'000;
constexpr std::uint32_t kFirstByteId = 3;
constexpr std::string_view kUnkToken = "<unk>";
constexpr std::string_view kBosToken = "<s>";
constexpr std::string_view kEosToken = "</s>";
constexpr std::string_view kWordBoundary = "▁";

std::string describe(const Json& value) {
    std::string result = value.dump();
    constexpr std::size_t kDiagnosticLimit = 256;
    if (result.size() > kDiagnosticLimit) {
        result.resize(kDiagnosticLimit);
        result += "...";
    }
    return result;
}

[[noreturn]] void reject(const std::filesystem::path& tokenizer_path,
                         std::string_view field,
                         std::string_view requirement,
                         std::string_view actual) {
    throw std::invalid_argument(
            "invalid tokenizer: " + tokenizer_path.string() + " field '" +
            std::string(field) + "' requires " + std::string(requirement) +
            "; actual " + std::string(actual));
}

[[noreturn]] void reject_value(const std::filesystem::path& path,
                               std::string_view field,
                               std::string_view requirement,
                               const Json& value) {
    reject(path, field, requirement, describe(value));
}

std::string child_field(std::string_view field, std::string_view child) {
    if (field.empty()) {
        return std::string(child);
    }
    return std::string(field) + "." + std::string(child);
}

std::string array_field(std::string_view field, std::size_t index) {
    return std::string(field) + "[" + std::to_string(index) + "]";
}

template <std::size_t N>
void require_exact_object_keys(
        const Json& value, const std::filesystem::path& path,
        std::string_view field,
        const std::array<std::string_view, N>& expected_keys) {
    if (!value.is_object()) {
        reject_value(path, field, "an object with the supported fields", value);
    }
    for (const auto& entry : value.items()) {
        const auto found = std::find(
                expected_keys.begin(), expected_keys.end(), entry.key());
        if (found == expected_keys.end()) {
            const std::string unknown = child_field(field, entry.key());
            reject(path, unknown, "a supported field", describe(entry.value()));
        }
    }
    for (const std::string_view key : expected_keys) {
        if (!value.contains(std::string(key))) {
            const std::string missing = child_field(field, key);
            reject(path, missing, "the required field", "<missing>");
        }
    }
}

std::string lookup_key(const Json& object, std::string_view field) {
    const std::string direct(field);
    if (object.contains(direct)) {
        return direct;
    }
    const std::size_t separator = field.rfind('.');
    if (separator != std::string_view::npos) {
        const std::string suffix(field.substr(separator + 1));
        if (object.contains(suffix)) {
            return suffix;
        }
    }
    return direct;
}

const Json& require_field(const Json& object,
                          const std::filesystem::path& path,
                          std::string_view field,
                          std::string_view requirement) {
    const std::string key = lookup_key(object, field);
    const auto found = object.find(key);
    if (found == object.end()) {
        reject(path, field, requirement, "<missing>");
    }
    return *found;
}

std::string require_string_value(const Json& value,
                                 const std::filesystem::path& path,
                                 std::string_view field,
                                 std::string_view expected) {
    const std::string requirement =
            "the string \"" + std::string(expected) + "\"";
    if (!value.is_string() || value.get<std::string>() != expected) {
        reject_value(path, field, requirement, value);
    }
    return value.get<std::string>();
}

std::uint32_t require_uint32_value(
        const Json& value, const std::filesystem::path& path,
        std::string_view field,
        std::string_view requirement =
                "a non-negative integer representable as uint32_t") {
    std::uint64_t raw = 0;
    if (value.is_number_unsigned()) {
        raw = value.get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const std::int64_t signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            reject_value(path, field, requirement, value);
        }
        raw = static_cast<std::uint64_t>(signed_value);
    } else {
        reject_value(path, field, requirement, value);
    }
    if (raw > std::numeric_limits<std::uint32_t>::max()) {
        reject_value(path, field, requirement, value);
    }
    return static_cast<std::uint32_t>(raw);
}

void require_null(const Json& object, const std::filesystem::path& path,
                  std::string_view field) {
    const Json& value = require_field(object, path, field, "null");
    if (!value.is_null()) {
        reject_value(path, field, "null", value);
    }
}

std::string require_string(const Json& object,
                           const std::filesystem::path& path,
                           std::string_view field,
                           std::string_view expected) {
    const Json& value = require_field(
            object, path, field,
            "the string \"" + std::string(expected) + "\"");
    return require_string_value(value, path, field, expected);
}

std::string require_any_string(const Json& object,
                               const std::filesystem::path& path,
                               std::string_view field) {
    const Json& value = require_field(object, path, field, "a string");
    if (!value.is_string()) {
        reject_value(path, field, "a string", value);
    }
    return value.get<std::string>();
}

bool require_bool(const Json& object, const std::filesystem::path& path,
                  std::string_view field, bool expected) {
    const Json& value = require_field(
            object, path, field,
            expected ? "the boolean true" : "the boolean false");
    if (!value.is_boolean() || value.get<bool>() != expected) {
        reject_value(path, field,
                     expected ? "the boolean true" : "the boolean false",
                     value);
    }
    return expected;
}

std::uint32_t require_uint32(const Json& object,
                             const std::filesystem::path& path,
                             std::string_view field,
                             std::string_view requirement =
                                     "a non-negative integer representable as uint32_t") {
    const Json& value = require_field(object, path, field, requirement);
    return require_uint32_value(value, path, field, requirement);
}


std::string read_tokenizer_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open tokenizer file " + path.string() +
                                 " for reading: " + std::strerror(errno));
    }
    std::string text{std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw std::runtime_error("cannot read tokenizer file " + path.string() +
                                 ": " + std::strerror(errno));
    }
    return text;
}

struct TokenPair {
    std::uint32_t first = 0;
    std::uint32_t second = 0;

    friend bool operator==(const TokenPair&, const TokenPair&) = default;
};

struct TokenPairHash {
    std::size_t operator()(const TokenPair& pair) const noexcept {
        const std::uint64_t packed =
                (static_cast<std::uint64_t>(pair.first) << 32) | pair.second;
        return static_cast<std::size_t>(packed ^ (packed >> 33));
    }
};

constexpr std::size_t kNoBpeIndex = std::numeric_limits<std::size_t>::max();

struct BpeNode {
    std::uint32_t id = 0;
    std::size_t previous = kNoBpeIndex;
    std::size_t next = kNoBpeIndex;
    std::size_t generation = 0;
    bool alive = true;
};

struct BpeHeapEntry {
    std::size_t rank = 0;
    std::size_t left = kNoBpeIndex;
    std::size_t right = kNoBpeIndex;
    std::size_t left_generation = 0;
    std::size_t right_generation = 0;
    std::uint32_t left_id = 0;
    std::uint32_t right_id = 0;
};

struct BpeHeapCompare {
    bool operator()(const BpeHeapEntry& lhs,
                    const BpeHeapEntry& rhs) const noexcept {
        if (lhs.rank != rhs.rank) {
            return lhs.rank > rhs.rank;
        }
        if (lhs.left != rhs.left) {
            return lhs.left > rhs.left;
        }
        return lhs.right > rhs.right;
    }
};


struct AddedTokenState {
    std::uint32_t id = 0;
    std::string content;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
    bool special = false;
};

struct TemplateItemState {
    bool special = false;
    std::string id;
    std::uint32_t type_id = 0;
};

struct SpecialTokenState {
    std::string id;
    std::vector<std::uint32_t> ids;
    std::vector<std::string> tokens;
};

struct NormalizerState {
    std::string prepend;
    std::string replace_pattern;
    std::string replace_content;
};

struct EncodingTables {
    const std::unordered_map<std::string, std::uint32_t>& vocabulary;
    const std::vector<std::string>& byte_pieces;
    const std::unordered_map<TokenPair, std::size_t, TokenPairHash>&
            merge_ranks;
    const std::unordered_map<TokenPair, std::uint32_t, TokenPairHash>&
            merge_result_ids;
    const NormalizerState& normalizer;
    bool fuse_unk = true;
    bool byte_fallback = true;
};

struct DecoderState {
    std::string replace_pattern;
    std::string replace_content;
    std::string strip_content;
    std::uint32_t strip_start = 0;
    std::uint32_t strip_stop = 0;
};

struct ParsedState {
    std::vector<std::string> vocabulary_by_id;
    std::unordered_map<std::string, std::uint32_t> vocabulary;
    std::vector<std::string> byte_pieces;
    std::vector<AddedTokenState> added_tokens;
    NormalizerState normalizer;
    std::vector<TemplateItemState> single_template;
    std::vector<TemplateItemState> pair_template;
    std::unordered_map<std::string, SpecialTokenState> special_tokens;
    DecoderState decoder;
    std::string unk_token;
    bool fuse_unk = true;
    bool byte_fallback = true;
    std::unordered_map<TokenPair, std::size_t, TokenPairHash> merge_ranks;
    std::unordered_map<TokenPair, std::uint32_t, TokenPairHash>
            merge_result_ids;
    std::vector<TokenPair> merges_by_rank;
};

std::string byte_piece(std::uint32_t byte) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string result = "<0x00>";
    result[3] = kHex[(byte >> 4) & 0x0F];
    result[4] = kHex[byte & 0x0F];
    return result;
}

void validate_added_tokens(const Json& document,
                           const std::filesystem::path& path,
                           ParsedState& state) {
    const Json& added = require_field(document, path, "added_tokens",
                                       "an array of exactly three entries");
    if (!added.is_array() || added.size() != 3) {
        reject_value(path, "added_tokens", "an array of exactly three entries",
                     added);
    }

    constexpr std::array<std::string_view, 3> expected_content = {
            kUnkToken, kBosToken, kEosToken};
    constexpr std::array<std::uint32_t, 3> expected_ids = {kUnkId, kBosId,
                                                            kEosId};
    // `special` is kept separately in the array so that the expected list stays
    // readable beside the artifact's seven-field entries.
    constexpr std::array<std::string_view, 7> all_keys = {
            "id", "content", "single_word", "lstrip", "rstrip", "normalized",
            "special"};

    state.added_tokens.reserve(added.size());
    for (std::size_t index = 0; index < added.size(); ++index) {
        const std::string field = array_field("added_tokens", index);
        const Json& entry = added.at(index);
        require_exact_object_keys(entry, path, field, all_keys);
        const std::uint32_t id = require_uint32(
                entry, path, child_field(field, "id"),
                "the integer " + std::to_string(expected_ids[index]));
        if (id != expected_ids[index]) {
            reject_value(path, child_field(field, "id"),
                         "the integer " + std::to_string(expected_ids[index]),
                         entry.at("id"));
        }
        const std::string content = require_string(
                entry, path, child_field(field, "content"),
                expected_content[index]);
        const bool single_word =
                require_bool(entry, path, child_field(field, "single_word"), false);
        const bool lstrip =
                require_bool(entry, path, child_field(field, "lstrip"), false);
        const bool rstrip =
                require_bool(entry, path, child_field(field, "rstrip"), false);
        const bool normalized = require_bool(
                entry, path, child_field(field, "normalized"), false);
        const bool special =
                require_bool(entry, path, child_field(field, "special"), true);
        state.added_tokens.push_back(
                AddedTokenState{id, content, single_word, lstrip, rstrip,
                                normalized, special});
    }
}

void validate_normalizer(const Json& document,
                         const std::filesystem::path& path,
                         ParsedState& state) {
    const Json& normalizer = require_field(
            document, path, "normalizer", "the supported normalizer object");
    constexpr std::array<std::string_view, 2> normalizer_keys = {"type",
                                                                  "normalizers"};
    require_exact_object_keys(normalizer, path, "normalizer", normalizer_keys);
    require_string(normalizer, path, "normalizer.type", "Sequence");
    const Json& normalizers = require_field(
            normalizer, path, "normalizer.normalizers", "an array of two normalizers");
    if (!normalizers.is_array() || normalizers.size() != 2) {
        reject_value(path, "normalizer.normalizers", "an array of two normalizers",
                     normalizers);
    }

    const Json& prepend = normalizers.at(0);
    constexpr std::array<std::string_view, 2> prepend_keys = {"type", "prepend"};
    require_exact_object_keys(prepend, path, "normalizer.normalizers[0]",
                              prepend_keys);
    require_string(prepend, path, "normalizer.normalizers[0].type", "Prepend");
    state.normalizer.prepend = require_string(
            prepend, path, "normalizer.normalizers[0].prepend", kWordBoundary);

    const Json& replace = normalizers.at(1);
    constexpr std::array<std::string_view, 3> replace_keys = {
            "type", "pattern", "content"};
    require_exact_object_keys(replace, path, "normalizer.normalizers[1]",
                              replace_keys);
    require_string(replace, path, "normalizer.normalizers[1].type", "Replace");
    const Json& pattern = require_field(replace, path,
                                        "normalizer.normalizers[1].pattern",
                                        "the String pattern object");
    constexpr std::array<std::string_view, 1> pattern_keys = {"String"};
    require_exact_object_keys(pattern, path,
                              "normalizer.normalizers[1].pattern", pattern_keys);
    state.normalizer.replace_pattern = require_string(
            pattern, path, "normalizer.normalizers[1].pattern.String", " ");
    state.normalizer.replace_content = require_string(
            replace, path, "normalizer.normalizers[1].content", kWordBoundary);

    const Json& pre_tokenizer = require_field(
            document, path, "pre_tokenizer", "null");
    if (!pre_tokenizer.is_null()) {
        reject_value(path, "pre_tokenizer", "null", pre_tokenizer);
    }
}

TemplateItemState parse_template_item(const Json& item,
                                      const std::filesystem::path& path,
                                      std::string_view field,
                                      bool special) {
    if (special) {
        constexpr std::array<std::string_view, 1> outer_keys = {"SpecialToken"};
        require_exact_object_keys(item, path, field, outer_keys);
        const Json& token = item.at("SpecialToken");
        const std::string token_field = child_field(field, "SpecialToken");
        constexpr std::array<std::string_view, 2> token_keys = {"id", "type_id"};
        require_exact_object_keys(token, path, token_field, token_keys);
        const std::string id =
                require_string(token, path, child_field(token_field, "id"), kBosToken);
        const std::uint32_t type_id = require_uint32(
                token, path, child_field(token_field, "type_id"),
                "the integer 0 or 1");
        if (type_id > 1) {
            reject_value(path, child_field(token_field, "type_id"),
                         "the integer 0 or 1", token.at("type_id"));
        }
        return TemplateItemState{true, id, type_id};
    }

    constexpr std::array<std::string_view, 1> outer_keys = {"Sequence"};
    require_exact_object_keys(item, path, field, outer_keys);
    const Json& sequence = item.at("Sequence");
    const std::string sequence_field = child_field(field, "Sequence");
    constexpr std::array<std::string_view, 2> sequence_keys = {"id", "type_id"};
    require_exact_object_keys(sequence, path, sequence_field, sequence_keys);
    const std::string id =
            require_any_string(sequence, path, child_field(sequence_field, "id"));
    if (id != "A" && id != "B") {
        reject_value(path, child_field(sequence_field, "id"),
                     "the string \"A\" or \"B\"", sequence.at("id"));
    }
    const std::uint32_t type_id = require_uint32(
            sequence, path, child_field(sequence_field, "type_id"),
            "the integer 0 or 1");
    if (type_id > 1) {
        reject_value(path, child_field(sequence_field, "type_id"),
                     "the integer 0 or 1", sequence.at("type_id"));
    }
    return TemplateItemState{false, id, type_id};
}

void validate_post_processor(const Json& document,
                             const std::filesystem::path& path,
                             ParsedState& state) {
    const Json& post_processor = require_field(
            document, path, "post_processor", "the TemplateProcessing object");
    constexpr std::array<std::string_view, 4> post_keys = {
            "type", "single", "pair", "special_tokens"};
    require_exact_object_keys(post_processor, path, "post_processor", post_keys);
    require_string(post_processor, path, "post_processor.type",
                   "TemplateProcessing");

    const Json& single = require_field(post_processor, path, "post_processor.single",
                                       "the single template array");
    if (!single.is_array() || single.size() != 2) {
        reject_value(path, "post_processor.single", "an array of two template items",
                     single);
    }
    const Json& pair = require_field(post_processor, path, "post_processor.pair",
                                     "the pair template array");
    if (!pair.is_array() || pair.size() != 4) {
        reject_value(path, "post_processor.pair", "an array of four template items",
                     pair);
    }

    state.single_template.reserve(single.size());
    state.single_template.push_back(parse_template_item(
            single.at(0), path, "post_processor.single[0]", true));
    state.single_template.push_back(parse_template_item(
            single.at(1), path, "post_processor.single[1]", false));
    if (!state.single_template[0].special ||
        state.single_template[0].id != kBosToken ||
        state.single_template[0].type_id != 0 || state.single_template[1].special ||
        state.single_template[1].id != "A" ||
        state.single_template[1].type_id != 0) {
        reject(path, "post_processor.single", "the exact BOS/$A template", describe(single));
    }

    state.pair_template.reserve(pair.size());
    state.pair_template.push_back(parse_template_item(
            pair.at(0), path, "post_processor.pair[0]", true));
    state.pair_template.push_back(parse_template_item(
            pair.at(1), path, "post_processor.pair[1]", false));
    state.pair_template.push_back(parse_template_item(
            pair.at(2), path, "post_processor.pair[2]", true));
    state.pair_template.push_back(parse_template_item(
            pair.at(3), path, "post_processor.pair[3]", false));
    if (!state.pair_template[0].special ||
        state.pair_template[0].id != kBosToken ||
        state.pair_template[0].type_id != 0 || state.pair_template[1].special ||
        state.pair_template[1].id != "A" || state.pair_template[1].type_id != 0 ||
        !state.pair_template[2].special ||
        state.pair_template[2].id != kBosToken ||
        state.pair_template[2].type_id != 1 || state.pair_template[3].special ||
        state.pair_template[3].id != "B" || state.pair_template[3].type_id != 1) {
        reject(path, "post_processor.pair", "the exact BOS/$A/BOS/$B template", describe(pair));
    }

    const Json& special_tokens = require_field(
            post_processor, path, "post_processor.special_tokens",
            "the exact <s> special-token table");
    constexpr std::array<std::string_view, 1> special_table_keys = {kBosToken};
    require_exact_object_keys(special_tokens, path, "post_processor.special_tokens",
                              special_table_keys);
    const Json& bos = special_tokens.at(std::string(kBosToken));
    constexpr std::array<std::string_view, 3> bos_keys = {"id", "ids", "tokens"};
    require_exact_object_keys(bos, path, "post_processor.special_tokens.<s>",
                              bos_keys);
    SpecialTokenState bos_state;
    bos_state.id = require_string(
            bos, path, "post_processor.special_tokens.<s>.id", kBosToken);
    const Json& ids = require_field(bos, path, "post_processor.special_tokens.<s>.ids",
                                    "an array containing integer 1");
    if (!ids.is_array() || ids.size() != 1) {
        reject_value(path, "post_processor.special_tokens.<s>.ids",
                     "an array containing integer 1", ids);
    }
    const Json& id_entry = ids.at(0);
    if (!id_entry.is_number_unsigned() ||
        id_entry.get<std::uint64_t>() != kBosId) {
        reject_value(path, "post_processor.special_tokens.<s>.ids",
                     "an array containing integer 1", ids);
    }
    bos_state.ids.push_back(kBosId);
    const Json& tokens = require_field(
            bos, path, "post_processor.special_tokens.<s>.tokens",
            "an array containing \"<s>\"");
    if (!tokens.is_array() || tokens.size() != 1 || !tokens.at(0).is_string() ||
        tokens.at(0).get<std::string>() != kBosToken) {
        reject_value(path, "post_processor.special_tokens.<s>.tokens",
                     "an array containing \"<s>\"", tokens);
    }
    bos_state.tokens.emplace_back(kBosToken);
    state.special_tokens.emplace(std::string(kBosToken), std::move(bos_state));
}

void validate_decoder(const Json& document,
                      const std::filesystem::path& path,
                      ParsedState& state) {
    const Json& decoder = require_field(document, path, "decoder",
                                        "the exact decoder sequence");
    constexpr std::array<std::string_view, 2> decoder_keys = {"type", "decoders"};
    require_exact_object_keys(decoder, path, "decoder", decoder_keys);
    require_string(decoder, path, "decoder.type", "Sequence");
    const Json& decoders = require_field(decoder, path, "decoder.decoders",
                                         "an array of four decoders");
    if (!decoders.is_array() || decoders.size() != 4) {
        reject_value(path, "decoder.decoders", "an array of four decoders", decoders);
    }

    const Json& replace = decoders.at(0);
    constexpr std::array<std::string_view, 3> replace_keys = {
            "type", "pattern", "content"};
    require_exact_object_keys(replace, path, "decoder.decoders[0]", replace_keys);
    require_string(replace, path, "decoder.decoders[0].type", "Replace");
    const Json& pattern = require_field(replace, path, "decoder.decoders[0].pattern",
                                        "the String pattern object");
    constexpr std::array<std::string_view, 1> pattern_keys = {"String"};
    require_exact_object_keys(pattern, path, "decoder.decoders[0].pattern",
                              pattern_keys);
    state.decoder.replace_pattern = require_string(
            pattern, path, "decoder.decoders[0].pattern.String", kWordBoundary);
    state.decoder.replace_content = require_string(
            replace, path, "decoder.decoders[0].content", " ");

    const Json& byte_fallback = decoders.at(1);
    constexpr std::array<std::string_view, 1> byte_keys = {"type"};
    require_exact_object_keys(byte_fallback, path, "decoder.decoders[1]", byte_keys);
    require_string(byte_fallback, path, "decoder.decoders[1].type", "ByteFallback");

    const Json& fuse = decoders.at(2);
    require_exact_object_keys(fuse, path, "decoder.decoders[2]", byte_keys);
    require_string(fuse, path, "decoder.decoders[2].type", "Fuse");

    const Json& strip = decoders.at(3);
    constexpr std::array<std::string_view, 4> strip_keys = {
            "type", "content", "start", "stop"};
    require_exact_object_keys(strip, path, "decoder.decoders[3]", strip_keys);
    require_string(strip, path, "decoder.decoders[3].type", "Strip");
    state.decoder.strip_content = require_string(
            strip, path, "decoder.decoders[3].content", " ");
    state.decoder.strip_start = require_uint32(
            strip, path, "decoder.decoders[3].start", "the integer 1");
    if (state.decoder.strip_start != 1) {
        reject_value(path, "decoder.decoders[3].start", "the integer 1",
                     strip.at("start"));
    }
    state.decoder.strip_stop = require_uint32(
            strip, path, "decoder.decoders[3].stop", "the integer 0");
    if (state.decoder.strip_stop != 0) {
        reject_value(path, "decoder.decoders[3].stop", "the integer 0",
                     strip.at("stop"));
    }
}

void validate_vocabulary_and_merges(const Json& document,
                                    const std::filesystem::path& path,
                                    ParsedState& state) {
    const Json& model = require_field(document, path, "model",
                                      "the exact BPE model object");
    constexpr std::array<std::string_view, 9> model_keys = {
            "type", "dropout", "unk_token", "continuing_subword_prefix",
            "end_of_word_suffix", "fuse_unk", "byte_fallback", "vocab", "merges"};
    require_exact_object_keys(model, path, "model", model_keys);
    require_string(model, path, "model.type", "BPE");
    require_null(model, path, "model.dropout");
    state.unk_token = require_string(model, path, "model.unk_token", kUnkToken);
    require_null(model, path, "model.continuing_subword_prefix");
    require_null(model, path, "model.end_of_word_suffix");
    state.fuse_unk = require_bool(model, path, "model.fuse_unk", true);
    state.byte_fallback = require_bool(model, path, "model.byte_fallback", true);

    const Json& vocab = require_field(model, path, "model.vocab",
                                      "an object with exactly 32000 entries");
    if (!vocab.is_object() || vocab.size() != kVocabularySize) {
        reject_value(path, "model.vocab", "an object with exactly 32000 entries",
                     vocab);
    }

    state.vocabulary_by_id.resize(kVocabularySize);
    state.vocabulary.reserve(vocab.size());
    std::vector<bool> seen_ids(kVocabularySize, false);
    for (const auto& entry : vocab.items()) {
        const std::string field = child_field("model.vocab", entry.key());
        if (entry.key().empty()) {
            reject(path, field, "a non-empty unique token string", "\"\"");
        }
        const std::uint32_t id = require_uint32_value(
                entry.value(), path, field,
                "an integer ID in the contiguous range 0..31999");
        if (id >= kVocabularySize) {
            reject_value(path, field, "an integer ID in the contiguous range 0..31999",
                         entry.value());
        }
        if (seen_ids[id]) {
            reject_value(path, field, "a unique integer ID", entry.value());
        }
        seen_ids[id] = true;
        const auto [inserted, ok] = state.vocabulary.emplace(entry.key(), id);
        if (!ok) {
            reject(path, field, "a unique token string", entry.key());
        }
        (void)inserted;
        state.vocabulary_by_id[id] = entry.key();
    }
    for (std::uint32_t id = 0; id < kVocabularySize; ++id) {
        if (!seen_ids[id]) {
            reject(path, "model.vocab", "contiguous unique IDs 0..31999",
                   "a missing ID " + std::to_string(id));
        }
    }

    const std::array<std::pair<std::uint32_t, std::string_view>, 3> special = {{
            {kUnkId, kUnkToken}, {kBosId, kBosToken}, {kEosId, kEosToken}}};
    for (const auto& [id, token] : special) {
        if (state.vocabulary_by_id[id] != token) {
            reject(path, "model.vocab", "the exact special token at ID " +
                                               std::to_string(id),
                   state.vocabulary_by_id[id]);
        }
    }
    state.byte_pieces.reserve(256);
    for (std::uint32_t byte = 0; byte <= 0xFF; ++byte) {
        const std::uint32_t id = kFirstByteId + byte;
        const std::string expected = byte_piece(byte);
        if (state.vocabulary_by_id[id] != expected) {
            reject(path, "model.vocab", "the exact byte piece " + expected +
                                               " at ID " + std::to_string(id),
                   state.vocabulary_by_id[id]);
        }
        state.byte_pieces.push_back(expected);
    }

    const Json& merges = require_field(model, path, "model.merges",
                                       "an array of valid BPE merge rows");
    if (!merges.is_array()) {
        reject_value(path, "model.merges", "an array of valid BPE merge rows",
                     merges);
    }
    state.merge_ranks.reserve(merges.size());
    state.merge_result_ids.reserve(merges.size());
    state.merges_by_rank.reserve(merges.size());
    std::size_t rank = 0;
    for (std::size_t index = 0; index < merges.size(); ++index) {
        const std::string field = array_field("model.merges", index);
        const Json& row = merges.at(index);
        if (!row.is_string()) {
            reject_value(path, field, "a string containing two vocabulary pieces",
                         row);
        }
        const std::string text = row.get<std::string>();
        const std::size_t separator = text.find(' ');
        if (separator == std::string::npos || separator == 0 ||
            separator + 1 >= text.size() ||
            text.find(' ', separator + 1) != std::string::npos) {
            reject(path, field,
                   "a pair of existing vocabulary pieces separated by one ASCII space",
                   text);
        }
        const std::string first = text.substr(0, separator);
        const std::string second = text.substr(separator + 1);
        const auto first_it = state.vocabulary.find(first);
        const auto second_it = state.vocabulary.find(second);
        if (first_it == state.vocabulary.end()) {
            reject(path, field, "a merge whose first operand exists in model.vocab",
                   text);
        }
        if (second_it == state.vocabulary.end()) {
            reject(path, field, "a merge whose second operand exists in model.vocab",
                   text);
        }
        if (second.size() > std::numeric_limits<std::size_t>::max() - first.size()) {
            throw std::overflow_error("tokenizer " + path.string() + " field '" +
                                      field + "' concatenated merge piece size overflows");
        }
        std::string result;
        result.reserve(first.size() + second.size());
        result += first;
        result += second;
        const auto result_it = state.vocabulary.find(result);
        if (result_it == state.vocabulary.end()) {
            reject(path, field,
                   "a merge whose concatenated result exists in model.vocab", text);
        }
        const TokenPair pair{first_it->second, second_it->second};
        if (state.merge_ranks.find(pair) != state.merge_ranks.end()) {
            reject(path, field, "a unique merge pair and rank", text);
        }
        state.merge_ranks.emplace(pair, rank);
        state.merge_result_ids.emplace(pair, result_it->second);
        state.merges_by_rank.push_back(pair);
        if (rank == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("tokenizer " + path.string() +
                                      " field 'model.merges' rank arithmetic overflows");
        }
        ++rank;
    }
}

ParsedState parse_tokenizer(const std::filesystem::path& path) {
    const std::string text = read_tokenizer_text(path);
    Json document;
    try {
        document = Json::parse(text);
    } catch (const Json::parse_error& error) {
        throw std::invalid_argument(
                "invalid tokenizer JSON: " + path.string() +
                " field '<document>' requires a valid JSON document; actual " +
                std::string(error.what()));
    }

    constexpr std::array<std::string_view, 9> top_level_keys = {
            "version", "truncation", "padding", "added_tokens", "normalizer",
            "pre_tokenizer", "post_processor", "decoder", "model"};
    require_exact_object_keys(document, path, "<document>", top_level_keys);
    require_string(document, path, "version", "1.0");
    require_null(document, path, "truncation");
    require_null(document, path, "padding");

    ParsedState state;
    validate_added_tokens(document, path, state);
    validate_normalizer(document, path, state);
    validate_post_processor(document, path, state);
    validate_decoder(document, path, state);
    validate_vocabulary_and_merges(document, path, state);
    return state;
}

bool is_byte_fallback_piece(std::string_view piece,
                            unsigned char& value) noexcept {
    if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
        piece[2] != 'x' || piece[5] != '>') {
        return false;
    }

    const auto hex_digit = [](char digit) noexcept -> unsigned char {
        if (digit >= '0' && digit <= '9') {
            return static_cast<unsigned char>(digit - '0');
        }
        if (digit >= 'A' && digit <= 'F') {
            return static_cast<unsigned char>(digit - 'A' + 10);
        }
        return 0xFF;
    };
    const unsigned char high = hex_digit(piece[3]);
    const unsigned char low = hex_digit(piece[4]);
    if (high == 0xFF || low == 0xFF) {
        return false;
    }
    value = static_cast<unsigned char>((high << 4) | low);
    return true;
}

std::size_t checked_decode_size_add(std::size_t current,
                                    std::size_t addition) {
    if (addition > std::numeric_limits<std::size_t>::max() - current) {
        throw std::overflow_error("tokenizer decode output size overflows");
    }
    return current + addition;
}

std::size_t decoded_piece_size(std::string_view piece,
                               const DecoderState& decoder) {
    unsigned char byte = 0;
    if (is_byte_fallback_piece(piece, byte)) {
        return 1;
    }

    const std::string_view pattern = decoder.replace_pattern;
    if (pattern.empty()) {
        return piece.size();
    }

    std::size_t total = 0;
    std::size_t cursor = 0;
    while (true) {
        const std::size_t match = piece.find(pattern, cursor);
        if (match == std::string_view::npos) {
            return checked_decode_size_add(total, piece.size() - cursor);
        }
        total = checked_decode_size_add(total, match - cursor);
        total = checked_decode_size_add(total, decoder.replace_content.size());
        cursor = match + pattern.size();
    }
}

void append_decoded_piece(std::string& output, std::string_view piece,
                          const DecoderState& decoder) {
    unsigned char byte = 0;
    if (is_byte_fallback_piece(piece, byte)) {
        output.push_back(static_cast<char>(byte));
        return;
    }

    const std::string_view pattern = decoder.replace_pattern;
    if (pattern.empty()) {
        output.append(piece.data(), piece.size());
        return;
    }

    std::size_t cursor = 0;
    while (true) {
        const std::size_t match = piece.find(pattern, cursor);
        if (match == std::string_view::npos) {
            output.append(piece.data() + cursor, piece.size() - cursor);
            return;
        }
        output.append(piece.data() + cursor, match - cursor);
        output.append(decoder.replace_content);
        cursor = match + pattern.size();
    }
}

bool is_skipped_special_id(std::uint32_t id) noexcept {
    return id == kUnkId || id == kBosId || id == kEosId;
}

}  // namespace

struct Tokenizer::Impl {
    explicit Impl(ParsedState state)
        : vocabulary_by_id(std::move(state.vocabulary_by_id)),
          vocabulary(std::move(state.vocabulary)),
          byte_pieces(std::move(state.byte_pieces)),
          added_tokens(std::move(state.added_tokens)),
          normalizer(std::move(state.normalizer)),
          single_template(std::move(state.single_template)),
          pair_template(std::move(state.pair_template)),
          special_tokens(std::move(state.special_tokens)),
          decoder(std::move(state.decoder)),
          unk_token(std::move(state.unk_token)),
          fuse_unk(state.fuse_unk),
          byte_fallback(state.byte_fallback),
          merge_ranks(std::move(state.merge_ranks)),
          merge_result_ids(std::move(state.merge_result_ids)),
          merges_by_rank(std::move(state.merges_by_rank)) {}

    std::vector<std::string> vocabulary_by_id;
    std::unordered_map<std::string, std::uint32_t> vocabulary;
    std::vector<std::string> byte_pieces;
    std::vector<AddedTokenState> added_tokens;
    NormalizerState normalizer;
    std::vector<TemplateItemState> single_template;
    std::vector<TemplateItemState> pair_template;
    std::unordered_map<std::string, SpecialTokenState> special_tokens;
    DecoderState decoder;
    std::string unk_token;
    bool fuse_unk = true;
    bool byte_fallback = true;
    std::unordered_map<TokenPair, std::size_t, TokenPairHash> merge_ranks;
    std::unordered_map<TokenPair, std::uint32_t, TokenPairHash>
            merge_result_ids;
    std::vector<TokenPair> merges_by_rank;
    std::vector<std::uint32_t> encode(
            std::string_view text, EncodeOptions options) const;

};

std::string utf8_byte_description(std::uint8_t byte) {
    constexpr char digits[] = "0123456789ABCDEF";
    std::string result = "0x00";
    result[2] = digits[(byte >> 4) & 0x0F];
    result[3] = digits[byte & 0x0F];
    return result;
}

[[noreturn]] void reject_encode_utf8(
        std::size_t offset, std::string_view expected,
        std::string_view actual) {
    throw std::invalid_argument(
            "Tokenizer::encode: invalid UTF-8 at byte " +
            std::to_string(offset) + "; expected " + std::string(expected) +
            "; actual " + std::string(actual));
}

void validate_encode_utf8(std::string_view text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto lead = static_cast<std::uint8_t>(text[offset]);
        std::size_t length = 0;
        std::string_view expected_lead;
        if (lead <= 0x7F) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
            expected_lead = "a valid 2-byte UTF-8 sequence";
        } else if (lead == 0xE0) {
            length = 3;
            expected_lead =
                    "a 3-byte UTF-8 sequence with second byte 0xA0..0xBF";
        } else if ((lead >= 0xE1 && lead <= 0xEC) || lead == 0xED ||
                   (lead >= 0xEE && lead <= 0xEF)) {
            length = 3;
            expected_lead = "a valid 3-byte UTF-8 sequence";
        } else if (lead == 0xF0) {
            length = 4;
            expected_lead =
                    "a 4-byte UTF-8 sequence with second byte 0x90..0xBF";
        } else if (lead >= 0xF1 && lead <= 0xF3) {
            length = 4;
            expected_lead = "a valid 4-byte UTF-8 sequence";
        } else if (lead == 0xF4) {
            length = 4;
            expected_lead =
                    "a 4-byte UTF-8 sequence with second byte 0x80..0x8F";
        } else {
            reject_encode_utf8(offset, "a UTF-8 leading byte",
                                utf8_byte_description(lead));
        }

        if (text.size() - offset < length) {
            reject_encode_utf8(
                    text.size(), expected_lead.empty() ? "continuation bytes"
                                                        : expected_lead,
                    "end-of-input");
        }

        const auto second = [&]() -> std::uint8_t {
            if (length == 1) {
                return 0;
            }
            return static_cast<std::uint8_t>(text[offset + 1]);
        }();
        if (length > 1) {
            std::uint8_t second_min = 0x80;
            std::uint8_t second_max = 0xBF;
            if (lead == 0xE0) {
                second_min = 0xA0;
            } else if (lead == 0xED) {
                second_max = 0x9F;
            } else if (lead == 0xF0) {
                second_min = 0x90;
            } else if (lead == 0xF4) {
                second_max = 0x8F;
            }
            if (second < second_min || second > second_max) {
                const std::string expected =
                        "a continuation byte in " +
                        utf8_byte_description(second_min) + ".." +
                        utf8_byte_description(second_max);
                reject_encode_utf8(offset + 1, expected,
                                   utf8_byte_description(second));
            }
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte =
                    static_cast<std::uint8_t>(text[offset + index]);
            if (byte < 0x80 || byte > 0xBF) {
                reject_encode_utf8(offset + index, "a continuation byte",
                                   utf8_byte_description(byte));
            }
        }
        offset += length;
    }
}

std::size_t encode_checked_add(
        std::size_t left, std::size_t right, std::string_view context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(
                "Tokenizer::encode: " + std::string(context) +
                " size overflows");
    }
    return left + right;
}

std::size_t encode_checked_mul(
        std::size_t left, std::size_t right, std::string_view context) {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
                "Tokenizer::encode: " + std::string(context) +
                " size overflows");
    }
    return left * right;
}

template <typename T>
void encode_checked_reserve(
        std::vector<T>& values, std::size_t size, std::string_view context) {
    if (size > values.max_size()) {
        throw std::overflow_error(
                "Tokenizer::encode: " + std::string(context) +
                " size overflows");
    }
    values.reserve(size);
}

void encode_checked_append(
        std::string& target, std::string_view value,
        std::string_view context) {
    if (value.size() > target.max_size() - target.size()) {
        throw std::overflow_error(
                "Tokenizer::encode: " + std::string(context) +
                " size overflows");
    }
    target.append(value);
}

void encode_checked_push(
        std::vector<std::uint32_t>& result, std::uint32_t id) {
    if (result.size() >= result.max_size()) {
        throw std::overflow_error(
                "Tokenizer::encode: result vector size overflows");
    }
    result.push_back(id);
}

void append_bpe_node(
        std::vector<BpeNode>& nodes, std::uint32_t id) {
    if (nodes.size() >= nodes.max_size()) {
        throw std::overflow_error(
                "Tokenizer::encode: BPE symbol count overflows");
    }
    const std::size_t index = nodes.size();
    nodes.push_back(BpeNode{id, index == 0 ? kNoBpeIndex : index - 1,
                            kNoBpeIndex, 0, true});
    if (index != 0) {
        nodes[index - 1].next = index;
    }
}

std::size_t encode_utf8_scalar_length(
        std::string_view text, std::size_t offset) {
    if (offset >= text.size()) {
        throw std::logic_error(
                "Tokenizer::encode: normalized UTF-8 offset is out of range");
    }
    const auto lead = static_cast<std::uint8_t>(text[offset]);
    if (lead <= 0x7F) {
        return 1;
    }
    if (lead <= 0xDF) {
        return 2;
    }
    if (lead <= 0xEF) {
        return 3;
    }
    if (lead <= 0xF4) {
        return 4;
    }
    throw std::logic_error(
            "Tokenizer::encode: normalized text is not valid UTF-8");
}

std::string normalize_encode_span(
        std::string_view plain, const NormalizerState& normalizer) {
    const std::size_t initial_size = encode_checked_add(
            plain.size(), normalizer.prepend.size(), "normalized input");
    std::string normalized;
    if (initial_size > normalized.max_size()) {
        throw std::overflow_error(
                "Tokenizer::encode: normalized input size overflows");
    }
    normalized.reserve(initial_size);
    encode_checked_append(normalized, normalizer.prepend, "normalized input");

    const std::string_view pattern = normalizer.replace_pattern;
    const std::string_view replacement = normalizer.replace_content;
    std::size_t offset = 0;
    while (offset < plain.size()) {
        if (!pattern.empty() && pattern.size() <= plain.size() - offset &&
            plain.compare(offset, pattern.size(), pattern) == 0) {
            encode_checked_append(normalized, replacement, "normalized input");
            offset += pattern.size();
        } else {
            encode_checked_append(
                    normalized, plain.substr(offset, 1), "normalized input");
            ++offset;
        }
    }
    return normalized;
}

void append_encode_initial_symbols(
        std::string_view normalized, const EncodingTables& tables,
        std::vector<BpeNode>& nodes) {
    encode_checked_reserve(nodes, normalized.size(), "BPE symbol");
    std::size_t offset = 0;
    while (offset < normalized.size()) {
        const std::size_t length =
                encode_utf8_scalar_length(normalized, offset);
        const std::string_view scalar = normalized.substr(offset, length);
        const auto found = tables.vocabulary.find(std::string(scalar));
        if (found != tables.vocabulary.end()) {
            append_bpe_node(nodes, found->second);
        } else if (tables.byte_fallback) {
            for (std::size_t index = 0; index < length; ++index) {
                const std::size_t byte = static_cast<std::uint8_t>(
                        normalized[offset + index]);
                if (byte >= tables.byte_pieces.size()) {
                    throw std::logic_error(
                            "Tokenizer::encode: byte fallback table is incomplete");
                }
                const auto byte_found =
                        tables.vocabulary.find(tables.byte_pieces[byte]);
                if (byte_found == tables.vocabulary.end()) {
                    throw std::logic_error(
                            "Tokenizer::encode: byte fallback piece is missing");
                }
                append_bpe_node(nodes, byte_found->second);
            }
        } else if (!tables.fuse_unk || nodes.empty() ||
                   nodes.back().id != kUnkId) {
            append_bpe_node(nodes, kUnkId);
        }
        offset += length;
    }
}

void apply_encode_bpe(
        std::vector<BpeNode>& nodes, const EncodingTables& tables) {
    if (nodes.empty()) {
        return;
    }

    const std::size_t heap_push_limit =
            encode_checked_mul(nodes.size(), 3, "BPE merge heap");
    std::vector<BpeHeapEntry> heap_storage;
    encode_checked_reserve(
            heap_storage, heap_push_limit, "BPE merge heap");
    std::priority_queue<BpeHeapEntry, std::vector<BpeHeapEntry>,
                        BpeHeapCompare>
            heap(BpeHeapCompare{}, std::move(heap_storage));
    std::size_t heap_pushes = 0;

    const auto queue_pair = [&](std::size_t left_index,
                                std::size_t right_index) {
        if (left_index == kNoBpeIndex || right_index == kNoBpeIndex) {
            return;
        }
        if (left_index >= nodes.size() || right_index >= nodes.size()) {
            throw std::logic_error(
                    "Tokenizer::encode: BPE linked-symbol index is invalid");
        }
        const BpeNode& left = nodes[left_index];
        const BpeNode& right = nodes[right_index];
        if (!left.alive || !right.alive || left.next != right_index ||
            right.previous != left_index) {
            return;
        }
        const TokenPair pair{left.id, right.id};
        const auto rank = tables.merge_ranks.find(pair);
        if (rank == tables.merge_ranks.end()) {
            return;
        }
        if (tables.merge_result_ids.find(pair) ==
            tables.merge_result_ids.end()) {
            throw std::logic_error(
                    "Tokenizer::encode: validated BPE merge has no result ID");
        }
        if (heap_pushes >= heap_push_limit) {
            throw std::overflow_error(
                    "Tokenizer::encode: BPE merge heap entry count overflows");
        }
        ++heap_pushes;
        heap.push(BpeHeapEntry{rank->second, left_index, right_index,
                               left.generation, right.generation, left.id,
                               right.id});
    };

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        queue_pair(index, nodes[index].next);
    }

    while (!heap.empty()) {
        const BpeHeapEntry entry = heap.top();
        heap.pop();
        if (entry.left >= nodes.size() || entry.right >= nodes.size()) {
            continue;
        }
        BpeNode& left = nodes[entry.left];
        BpeNode& right = nodes[entry.right];
        if (!left.alive || !right.alive || left.next != entry.right ||
            right.previous != entry.left ||
            left.generation != entry.left_generation ||
            right.generation != entry.right_generation ||
            left.id != entry.left_id || right.id != entry.right_id) {
            continue;
        }
        const TokenPair pair{left.id, right.id};
        const auto result = tables.merge_result_ids.find(pair);
        if (result == tables.merge_result_ids.end()) {
            throw std::logic_error(
                    "Tokenizer::encode: validated BPE merge result disappeared");
        }
        if (left.generation == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                    "Tokenizer::encode: BPE generation arithmetic overflows");
        }
        const std::size_t next = right.next;
        const std::size_t previous = left.previous;
        left.id = result->second;
        left.next = next;
        ++left.generation;
        right.alive = false;
        right.previous = kNoBpeIndex;
        right.next = kNoBpeIndex;
        if (next != kNoBpeIndex) {
            if (next >= nodes.size()) {
                throw std::logic_error(
                        "Tokenizer::encode: BPE successor index is invalid");
            }
            nodes[next].previous = entry.left;
        }
        queue_pair(previous, entry.left);
        queue_pair(entry.left, next);
    }
}

void append_encode_plain_span(
        std::string_view plain, const EncodingTables& tables,
        std::vector<std::uint32_t>& result) {
    if (plain.empty()) {
        return;
    }
    const std::string normalized =
            normalize_encode_span(plain, tables.normalizer);
    std::vector<BpeNode> nodes;
    append_encode_initial_symbols(normalized, tables, nodes);
    apply_encode_bpe(nodes, tables);
    std::size_t index = nodes.empty() ? kNoBpeIndex : 0;
    std::size_t visited = 0;
    while (index != kNoBpeIndex) {
        if (index >= nodes.size() || !nodes[index].alive ||
            visited >= nodes.size()) {
            throw std::logic_error(
                    "Tokenizer::encode: BPE linked-symbol traversal is invalid");
        }
        encode_checked_push(result, nodes[index].id);
        ++visited;
        index = nodes[index].next;
    }
}

std::vector<std::uint32_t> Tokenizer::Impl::encode(
        std::string_view text, EncodeOptions options) const {
    validate_encode_utf8(text);

    std::vector<std::uint32_t> result;
    const std::size_t reserve_size =
            encode_checked_add(text.size(), 2, "result");
    encode_checked_reserve(result, reserve_size, "result");

    const EncodingTables tables{vocabulary, byte_pieces, merge_ranks,
                                merge_result_ids, normalizer, fuse_unk,
                                byte_fallback};

    if (options.add_special_tokens) {
        bool sequence_seen = false;
        for (const TemplateItemState& item : single_template) {
            if (!item.special) {
                if (item.id != "A") {
                    throw std::logic_error(
                            "Tokenizer::encode: single template sequence is not A");
                }
                sequence_seen = true;
                break;
            }
            const auto special = special_tokens.find(item.id);
            if (special == special_tokens.end()) {
                throw std::logic_error(
                        "Tokenizer::encode: single template special token is missing");
            }
            for (const std::uint32_t id : special->second.ids) {
                encode_checked_push(result, id);
            }
        }
        if (!sequence_seen) {
            throw std::logic_error(
                    "Tokenizer::encode: single template has no sequence");
        }
    }

    std::size_t plain_start = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        std::size_t match_index = kNoBpeIndex;
        std::size_t match_length = 0;
        std::uint32_t match_id = 0;
        for (const AddedTokenState& token : added_tokens) {
            if (token.content.size() <= text.size() - offset &&
                text.compare(offset, token.content.size(), token.content) == 0 &&
                token.content.size() > match_length) {
                match_index = offset;
                match_length = token.content.size();
                match_id = token.id;
            }
        }
        if (match_index == kNoBpeIndex) {
            ++offset;
            continue;
        }
        append_encode_plain_span(
                text.substr(plain_start, match_index - plain_start),
                tables, result);
        encode_checked_push(result, match_id);
        offset = encode_checked_add(offset, match_length, "input");
        plain_start = offset;
    }
    append_encode_plain_span(text.substr(plain_start), tables, result);
    return result;
}

std::vector<std::uint32_t> Tokenizer::encode(
        std::string_view text, EncodeOptions options) const {
    return impl_->encode(text, options);
}

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Tokenizer::~Tokenizer() = default;

std::uint32_t Tokenizer::bos_id() const noexcept {
    return kBosId;
}

std::uint32_t Tokenizer::eos_id() const noexcept {
    return kEosId;
}

std::uint32_t Tokenizer::unk_id() const noexcept {
    return kUnkId;
}

std::uint32_t Tokenizer::tokenizer_pad_id() const noexcept {
    return kEosId;
}
std::string Tokenizer::decode(std::span<const std::uint32_t> ids,
                              DecodeOptions options) const {
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const std::uint32_t id = ids[index];
        if (id >= impl_->vocabulary_by_id.size()) {
            const std::size_t maximum = impl_->vocabulary_by_id.empty()
                    ? 0
                    : impl_->vocabulary_by_id.size() - 1;
            throw std::invalid_argument(
                    "tokenizer decode ID at index " + std::to_string(index) +
                    " is " + std::to_string(id) + "; valid range is 0.." +
                    std::to_string(maximum));
        }
    }

    std::size_t output_size = 0;
    for (const std::uint32_t id : ids) {
        if (options.skip_special_tokens && is_skipped_special_id(id)) {
            continue;
        }
        output_size = checked_decode_size_add(
                output_size,
                decoded_piece_size(impl_->vocabulary_by_id[id],
                                   impl_->decoder));
    }

    std::string result;
    if (output_size > result.max_size()) {
        throw std::overflow_error(
                "tokenizer decode output size exceeds string limit");
    }
    result.reserve(output_size);

    for (const std::uint32_t id : ids) {
        if (options.skip_special_tokens && is_skipped_special_id(id)) {
            continue;
        }
        append_decoded_piece(result, impl_->vocabulary_by_id[id],
                             impl_->decoder);
    }

    if (impl_->decoder.strip_start == 1 &&
        impl_->decoder.strip_stop == 0 &&
        impl_->decoder.strip_content == " " && !result.empty() &&
        result.front() == ' ') {
        result.erase(0, 1);
    }
    return result;
}


std::unique_ptr<Tokenizer> load_tokenizer(
        const std::filesystem::path& model_directory) {
    const std::filesystem::path tokenizer_path = model_directory / "tokenizer.json";
    ParsedState state = parse_tokenizer(tokenizer_path);
    auto impl = std::make_unique<Tokenizer::Impl>(std::move(state));
    return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(impl)));
}

}  // namespace iom
