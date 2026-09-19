#include "iom/tokenizer.hpp"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
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
        if (state.vocabulary.find(result) == state.vocabulary.end()) {
            reject(path, field,
                   "a merge whose concatenated result exists in model.vocab", text);
        }
        const TokenPair pair{first_it->second, second_it->second};
        if (state.merge_ranks.find(pair) != state.merge_ranks.end()) {
            reject(path, field, "a unique merge pair and rank", text);
        }
        state.merge_ranks.emplace(pair, rank);
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
    std::vector<TokenPair> merges_by_rank;
};

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

std::unique_ptr<Tokenizer> load_tokenizer(
        const std::filesystem::path& model_directory) {
    const std::filesystem::path tokenizer_path = model_directory / "tokenizer.json";
    ParsedState state = parse_tokenizer(tokenizer_path);
    auto impl = std::make_unique<Tokenizer::Impl>(std::move(state));
    return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(impl)));
}

}  // namespace iom
