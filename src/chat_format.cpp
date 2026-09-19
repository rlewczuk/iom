#include "iom/chat_format.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace iom {
namespace {

constexpr const char* kTokenizerConfigFileName = "tokenizer_config.json";
constexpr std::string_view kAddedTokenFields[] = {
        "content", "lstrip", "normalized", "rstrip", "single_word", "special"};
constexpr std::string_view kSupportedConfigKeys[] = {
        "added_tokens_decoder",
        "bos_token",
        "chat_template",
        "clean_up_tokenization_spaces",
        "eos_token",
        "legacy",
        "model_max_length",
        "pad_token",
        "padding_side",
        "sp_model_kwargs",
        "tokenizer_class",
        "unk_token",
        "use_default_system_prompt",
};

struct RoleAffix {
    std::string before_content;
    std::string after_content;
};

enum class TokenKind { text, output, block };

struct Token {
    TokenKind kind;
    std::string text;
    std::size_t offset;
};

enum class BlockKind { for_open, endfor, if_open, elif, else_block, endif, unknown };

struct BlockInfo {
    BlockKind kind;
    std::size_t offset;
};

enum class ReferenceKind { role, content, unsupported };

enum class ExpressionTermKind { literal, content, eos };

struct ExpressionTerm {
    ExpressionTermKind kind;
    std::string literal;
};

struct CompiledTemplate {
    std::array<RoleAffix, 3> role_affixes;
    std::string generation;
    std::string iteration_tail;
    std::string iteration_tail_after_generation;
    bool generation_inside_loop = false;
    std::string prefix;
    std::string post_loop;
    std::string suffix;
};

[[nodiscard]] std::string json_description(const nlohmann::json& value) {
    return value.dump();
}

[[noreturn]] void reject_field(const std::filesystem::path& config_path,
                              std::string_view field,
                              std::string_view expected,
                              std::string_view actual) {
    throw std::invalid_argument(
            "invalid tokenizer configuration: " + config_path.string()
            + " field '" + std::string(field) + "' requires "
            + std::string(expected) + "; actual " + std::string(actual));
}

[[noreturn]] void reject_template(const std::string& source_label,
                                  std::size_t offset,
                                  std::string_view expected,
                                  std::string_view actual) {
    throw std::invalid_argument(
            "invalid chat template: " + source_label + " at byte offset "
            + std::to_string(offset) + " requires " + std::string(expected)
            + "; actual " + std::string(actual));
}

[[nodiscard]] bool is_ascii_space(unsigned char value) noexcept {
    return value == static_cast<unsigned char>(' ')
           || value == static_cast<unsigned char>('\t')
           || value == static_cast<unsigned char>('\n')
           || value == static_cast<unsigned char>('\v')
           || value == static_cast<unsigned char>('\f');
}

[[nodiscard]] bool is_identifier_start(unsigned char value) noexcept {
    return (value >= static_cast<unsigned char>('a')
            && value <= static_cast<unsigned char>('z'))
           || (value >= static_cast<unsigned char>('A')
               && value <= static_cast<unsigned char>('Z'))
           || value == static_cast<unsigned char>('_');
}

[[nodiscard]] bool is_identifier_continue(unsigned char value) noexcept {
    return is_identifier_start(value)
           || (value >= static_cast<unsigned char>('0')
               && value <= static_cast<unsigned char>('9'));
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string() + ": "
                                 + std::strerror(errno));
    }
    std::string text{std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw std::runtime_error("cannot read " + path.string() + ": "
                                 + std::strerror(errno));
    }
    return text;
}

[[nodiscard]] const nlohmann::json& require_field(
        const nlohmann::json& document, const std::filesystem::path& path,
        std::string_view field, std::string_view expected) {
    const auto entry = document.find(std::string(field));
    if (entry == document.end()) {
        reject_field(path, field, expected, "<missing>");
    }
    return *entry;
}

void require_exact_keys(const nlohmann::json& document,
                        const std::filesystem::path& path,
                        std::string_view object_name,
                        const std::string_view* supported,
                        std::size_t supported_count) {
    if (!document.is_object()) {
        reject_field(path, object_name, "a JSON object with the supported keys",
                     json_description(document));
    }
    for (const auto& entry : document.items()) {
        bool supported_key = false;
        for (std::size_t i = 0; i < supported_count; ++i) {
            if (entry.key() == supported[i]) {
                supported_key = true;
                break;
            }
        }
        if (!supported_key) {
            reject_field(path, entry.key(), "a supported tokenizer key",
                         json_description(entry.value()));
        }
    }
    for (std::size_t i = 0; i < supported_count; ++i) {
        if (!document.contains(std::string(supported[i]))) {
            reject_field(path, supported[i], "a required field", "<missing>");
        }
    }
}

void require_string(const nlohmann::json& document,
                    const std::filesystem::path& path,
                    std::string_view field) {
    const auto& value = require_field(document, path, field, "a JSON string");
    if (!value.is_string()) {
        reject_field(path, field, "a JSON string", json_description(value));
    }
}

void require_exact_string(const nlohmann::json& document,
                          const std::filesystem::path& path,
                          std::string_view field, std::string_view expected) {
    const auto& value = require_field(document, path, field,
                                      "the string \"" + std::string(expected)
                                              + "\"");
    if (!value.is_string() || value.get<std::string>() != expected) {
        reject_field(path, field, "the string \"" + std::string(expected) + "\"",
                     json_description(value));
    }
}

void require_exact_bool(const nlohmann::json& document,
                        const std::filesystem::path& path,
                        std::string_view field, bool expected) {
    const std::string expected_text = expected ? "the boolean true"
                                               : "the boolean false";
    const auto& value = require_field(document, path, field, expected_text);
    if (!value.is_boolean() || value.get<bool>() != expected) {
        reject_field(path, field, expected_text, json_description(value));
    }
}

void require_exact_integer(const nlohmann::json& document,
                           const std::filesystem::path& path,
                           std::string_view field, std::uint64_t expected) {
    const std::string expected_text =
            "the integer " + std::to_string(expected);
    const auto& value = require_field(document, path, field, expected_text);
    bool matches = false;
    if (value.is_number_unsigned()) {
        matches = value.get<std::uint64_t>() == expected;
    } else if (value.is_number_integer()) {
        const auto actual = value.get<std::int64_t>();
        matches = actual >= 0 && static_cast<std::uint64_t>(actual) == expected;
    }
    if (!matches) {
        reject_field(path, field, expected_text, json_description(value));
    }
}

void validate_added_token(const nlohmann::json& decoder,
                          const std::filesystem::path& path,
                          std::string_view key,
                          std::string_view content) {
    const auto& value = require_field(decoder, path, key,
                                      "a token metadata object");
    require_exact_keys(value, path, "added_tokens_decoder[" + std::string(key)
                                      + "]", kAddedTokenFields,
                       std::size(kAddedTokenFields));
    require_exact_string(value, path, "content", content);
    require_exact_bool(value, path, "lstrip", false);
    require_exact_bool(value, path, "normalized", false);
    require_exact_bool(value, path, "rstrip", false);
    require_exact_bool(value, path, "single_word", false);
    require_exact_bool(value, path, "special", true);
}

void validate_metadata(const nlohmann::json& document,
                       const std::filesystem::path& path,
                       bool validate_default_template) {
    if (!document.is_object() || document.empty()) {
        reject_field(path, "<tokenizer_config.json>",
                     "a non-empty JSON object", json_description(document));
    }
    require_exact_keys(document, path, "<tokenizer_config.json>",
                       kSupportedConfigKeys,
                       std::size(kSupportedConfigKeys));

    const auto& decoder = require_field(document, path, "added_tokens_decoder",
                                        "the object with token ids 0, 1, and 2");
    constexpr std::string_view decoder_keys[] = {"0", "1", "2"};
    require_exact_keys(decoder, path, "added_tokens_decoder", decoder_keys,
                       std::size(decoder_keys));
    validate_added_token(decoder, path, "0", "<unk>");
    validate_added_token(decoder, path, "1", "<s>");
    validate_added_token(decoder, path, "2", "</s>");

    require_exact_string(document, path, "bos_token", "<s>");
    if (validate_default_template) {
        require_string(document, path, "chat_template");
    } else if (!document.contains("chat_template")) {
        reject_field(path, "chat_template", "a required field", "<missing>");
    }
    require_exact_bool(document, path, "clean_up_tokenization_spaces", false);
    require_exact_string(document, path, "eos_token", "</s>");
    require_exact_bool(document, path, "legacy", false);
    require_exact_integer(document, path, "model_max_length", 2048);
    require_exact_string(document, path, "pad_token", "</s>");
    require_exact_string(document, path, "padding_side", "right");
    const auto& kwargs = require_field(document, path, "sp_model_kwargs",
                                       "an empty JSON object");
    if (!kwargs.is_object() || !kwargs.empty()) {
        reject_field(path, "sp_model_kwargs", "an empty JSON object",
                     json_description(kwargs));
    }
    require_exact_string(document, path, "tokenizer_class", "LlamaTokenizer");
    require_exact_string(document, path, "unk_token", "<unk>");
    require_exact_bool(document, path, "use_default_system_prompt", false);
}

[[nodiscard]] std::string selected_template(
        const nlohmann::json& document, const std::filesystem::path& path,
        std::optional<std::string_view> template_override) {
    if (template_override.has_value()) {
        return std::string(*template_override);
    }
    const auto& value = require_field(document, path, "chat_template",
                                      "a UTF-8 string template");
    if (!value.is_string()) {
        reject_field(path, "chat_template", "a UTF-8 string template",
                     json_description(value));
    }
    return value.get<std::string>();
}

void validate_utf8_and_line_endings(std::string_view source,
                                    const std::string& source_label) {
    for (std::size_t offset = 0; offset < source.size();) {
        const auto byte = static_cast<unsigned char>(source[offset]);
        if (byte == static_cast<unsigned char>('\r')) {
            reject_template(source_label, offset, "LF line endings only",
                            "a raw CR byte");
        }
        if (byte <= 0x7F) {
            ++offset;
            continue;
        }
        std::size_t length = 0;
        std::uint32_t code_point = 0;
        if (byte >= 0xC2 && byte <= 0xDF) {
            length = 2;
            code_point = byte & 0x1F;
        } else if (byte >= 0xE0 && byte <= 0xEF) {
            length = 3;
            code_point = byte & 0x0F;
        } else if (byte >= 0xF0 && byte <= 0xF4) {
            length = 4;
            code_point = byte & 0x07;
        } else {
            reject_template(source_label, offset, "valid UTF-8", "an invalid lead byte");
        }
        if (offset + length > source.size()) {
            reject_template(source_label, offset, "valid UTF-8", "a truncated sequence");
        }
        for (std::size_t i = 1; i < length; ++i) {
            const auto continuation = static_cast<unsigned char>(source[offset + i]);
            if ((continuation & 0xC0) != 0x80) {
                reject_template(source_label, offset + i,
                                "valid UTF-8", "a non-continuation byte");
            }
            code_point = (code_point << 6) | (continuation & 0x3F);
        }
        if ((length == 3 && code_point < 0x800)
                || (length == 4 && code_point < 0x10000)
                || code_point > 0x10FFFF
                || (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            reject_template(source_label, offset, "valid UTF-8",
                            "an overlong, surrogate, or out-of-range sequence");
        }
        offset += length;
    }
}

[[nodiscard]] std::size_t find_tag_end(std::string_view source,
                                       std::size_t begin,
                                       std::string_view closing,
                                       const std::string& source_label) {
    char quote = '\0';
    bool escaped = false;
    for (std::size_t offset = begin; offset + 1 < source.size(); ++offset) {
        const char value = source[offset];
        if (quote != '\0') {
            if (escaped) {
                escaped = false;
            } else if (value == '\\') {
                escaped = true;
            } else if (value == quote) {
                quote = '\0';
            }
            continue;
        }
        if (value == '\'' || value == '"') {
            quote = value;
            continue;
        }
        if (source.compare(offset, closing.size(), closing) == 0) {
            return offset;
        }
    }
    reject_template(source_label, begin - 2, "a matched template tag", "an unterminated tag");
}

[[nodiscard]] std::vector<Token> tokenize_template(
        std::string_view source, const std::string& source_label) {
    std::vector<Token> tokens;
    std::size_t cursor = 0;
    while (cursor < source.size()) {
        const std::size_t open = source.find('{', cursor);
        if (open == std::string_view::npos) {
            tokens.push_back(Token{TokenKind::text,
                                   std::string(source.substr(cursor)), cursor});
            break;
        }
        const bool is_output = open + 1 < source.size()
                               && source[open + 1] == '{';
        const bool is_block = open + 1 < source.size()
                              && source[open + 1] == '%';
        const bool is_comment = open + 1 < source.size()
                                && source[open + 1] == '#';
        if (!is_output && !is_block) {
            if (is_comment) {
                reject_template(source_label, open, "no Jinja comments",
                                "a comment tag");
            }
            const std::size_t next_open = source.find('{', open + 1);
            const std::size_t text_end = next_open == std::string_view::npos
                                                 ? source.size()
                                                 : next_open;
            if (text_end > cursor) {
                tokens.push_back(Token{TokenKind::text,
                                       std::string(source.substr(cursor,
                                                                  text_end - cursor)),
                                       cursor});
            }
            cursor = text_end;
            continue;
        }
        if (open > cursor) {
            tokens.push_back(Token{TokenKind::text,
                                   std::string(source.substr(cursor, open - cursor)),
                                   cursor});
        }
        if (open + 2 < source.size() && source[open + 2] == '-') {
            reject_template(source_label, open, "ordinary Jinja delimiters",
                            "a whitespace-control opening delimiter");
        }
        const std::string_view closing = is_output ? "}}" : "%}";
        const std::size_t close =
                find_tag_end(source, open + 2, closing, source_label);
        if (close > open + 2 && source[close - 1] == '-') {
            reject_template(source_label, close - 1,
                            "ordinary Jinja delimiters",
                            "a whitespace-control closing delimiter");
        }
        tokens.push_back(Token{is_output ? TokenKind::output : TokenKind::block,
                               std::string(source.substr(open + 2,
                                                          close - (open + 2))),
                               open});
        cursor = close + 2;
    }

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].kind != TokenKind::block) {
            continue;
        }
        if (index > 0 && tokens[index - 1].kind == TokenKind::text) {
            std::string& previous = tokens[index - 1].text;
            const std::size_t line_break = previous.rfind('\n');
            const std::size_t line_start =
                    line_break == std::string::npos ? 0 : line_break + 1;
            bool line_leading = line_break != std::string::npos
                                || tokens[index - 1].offset == 0;
            for (std::size_t position = line_start;
                 line_leading && position < previous.size(); ++position) {
                line_leading = previous[position] == ' '
                               || previous[position] == '\t';
            }
            if (line_leading) {
                while (!previous.empty()
                       && (previous.back() == ' ' || previous.back() == '\t')) {
                    previous.pop_back();
                }
            }
        }
    }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].kind != TokenKind::block) {
            continue;
        }
        if (index + 1 < tokens.size()
                && tokens[index + 1].kind == TokenKind::text
                && !tokens[index + 1].text.empty()
                && tokens[index + 1].text.front() == '\n') {
            tokens[index + 1].text.erase(0, 1);
        }
    }
    return tokens;
}

class ExpressionParser {
public:
    ExpressionParser(std::string_view expression, std::size_t base_offset,
                     const std::string& source_label)
        : expression_(expression), base_offset_(base_offset),
          source_label_(source_label) {}

    void skip_space() noexcept {
        while (position_ < expression_.size()
               && is_ascii_space(static_cast<unsigned char>(
                       expression_[position_]))) {
            ++position_;
        }
    }

    [[nodiscard]] bool at_end() {
        skip_space();
        return position_ == expression_.size();
    }

    [[nodiscard]] std::string read_word() {
        skip_space();
        if (position_ == expression_.size()
            || !is_identifier_start(static_cast<unsigned char>(
                    expression_[position_]))) {
            fail("an identifier", "a non-identifier expression token");
        }
        const std::size_t begin = position_++;
        while (position_ < expression_.size()
               && is_identifier_continue(static_cast<unsigned char>(
                       expression_[position_]))) {
            ++position_;
        }
        return std::string(expression_.substr(begin, position_ - begin));
    }

    void expect_word(std::string_view expected) {
        const std::string actual = read_word();
        if (actual != expected) {
            fail("the identifier " + std::string(expected), actual);
        }
    }

    void expect_character(char expected) {
        skip_space();
        if (position_ == expression_.size() || expression_[position_] != expected) {
            fail(std::string("the character '") + expected + "'",
                 position_ == expression_.size()
                         ? "end of expression"
                         : std::string(1, expression_[position_]));
        }
        ++position_;
    }

    [[nodiscard]] std::string parse_string() {
        skip_space();
        if (position_ == expression_.size()
            || (expression_[position_] != '\''
                && expression_[position_] != '"')) {
            fail("a single- or double-quoted string", "a different expression term");
        }
        const char quote = expression_[position_++];
        std::string value;
        while (position_ < expression_.size()) {
            const char current = expression_[position_++];
            if (current == quote) {
                return value;
            }
            if (current == '\\') {
                if (position_ == expression_.size()) {
                    fail("a supported escape", "a trailing backslash");
                }
                const char escaped = expression_[position_++];
                switch (escaped) {
                    case '\\': value.push_back('\\'); break;
                    case '\'': value.push_back('\''); break;
                    case '"': value.push_back('"'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default:
                        fail("a supported escape: one of \\, \\', \", \\n, \\r, or \\t",
                             std::string("\\") + escaped);
                }
            } else {
                value.push_back(current);
            }
        }
        fail("a closing quote", "an unterminated string literal");
    }

    [[nodiscard]] ReferenceKind parse_reference() {
        const std::string variable = read_word();
        if (variable != "message") {
            fail("the variable message", variable);
        }
        skip_space();
        std::string field;
        if (position_ < expression_.size() && expression_[position_] == '.') {
            ++position_;
            field = read_word();
        } else {
            expect_character('[');
            field = parse_string();
            expect_character(']');
        }
        if (field == "role") {
            return ReferenceKind::role;
        }
        if (field == "content") {
            return ReferenceKind::content;
        }
        fail("the message field role or content", field);
    }

    void expect_operator(std::string_view expected) {
        skip_space();
        if (expression_.substr(position_, expected.size()) != expected) {
            fail("the operator " + std::string(expected),
                 position_ == expression_.size()
                         ? "end of expression"
                         : std::string(1, expression_[position_]));
        }
        position_ += expected.size();
    }

    void expect_end() {
        if (!at_end()) {
            fail("the end of the expression",
                 std::string(1, expression_[position_]));
        }
    }

    [[nodiscard]] std::size_t offset() const noexcept {
        return base_offset_ + position_;
    }

private:
    [[noreturn]] void fail(std::string expected, std::string actual) const {
        reject_template(source_label_, base_offset_ + position_, expected, actual);
    }

    std::string_view expression_;
    std::size_t base_offset_;
    const std::string& source_label_;
    std::size_t position_ = 0;
};

[[nodiscard]] BlockInfo classify_block(const Token& token) {
    if (token.kind != TokenKind::block) {
        return BlockInfo{BlockKind::unknown, token.offset};
    }
    std::size_t position = 0;
    while (position < token.text.size()
           && is_ascii_space(static_cast<unsigned char>(token.text[position]))) {
        ++position;
    }
    const std::size_t begin = position;
    if (position == token.text.size()
        || !is_identifier_start(static_cast<unsigned char>(token.text[position]))) {
        return BlockInfo{BlockKind::unknown, token.offset};
    }
    ++position;
    while (position < token.text.size()
           && is_identifier_continue(static_cast<unsigned char>(token.text[position]))) {
        ++position;
    }
    const std::string_view keyword(token.text.data() + begin, position - begin);
    if (keyword == "for") return BlockInfo{BlockKind::for_open, token.offset};
    if (keyword == "endfor") return BlockInfo{BlockKind::endfor, token.offset};
    if (keyword == "if") return BlockInfo{BlockKind::if_open, token.offset};
    if (keyword == "elif") return BlockInfo{BlockKind::elif, token.offset};
    if (keyword == "else") return BlockInfo{BlockKind::else_block, token.offset};
    if (keyword == "endif") return BlockInfo{BlockKind::endif, token.offset};
    return BlockInfo{BlockKind::unknown, token.offset};
}

void require_whitespace_text(const Token& token, const std::string& source_label) {
    for (unsigned char value : token.text) {
        if (!is_ascii_space(value)) {
            reject_template(source_label, token.offset, "whitespace literal text only",
                            token.text);
        }
    }
}

[[nodiscard]] std::size_t role_index(std::string_view role,
                                      const std::string& source_label,
                                      std::size_t offset) {
    if (role == "system") return 0;
    if (role == "user") return 1;
    if (role == "assistant") return 2;
    reject_template(source_label, offset,
                    "the role string system, user, or assistant", role);
}

void parse_simple_block(const Token& token, std::string_view keyword,
                        const std::string& source_label) {
    ExpressionParser parser(token.text, token.offset + 2, source_label);
    parser.expect_word(keyword);
    parser.expect_end();
}

void parse_for(const Token& token, const std::string& source_label) {
    ExpressionParser parser(token.text, token.offset + 2, source_label);
    parser.expect_word("for");
    parser.expect_word("message");
    parser.expect_word("in");
    parser.expect_word("messages");
    parser.expect_end();
}

[[nodiscard]] std::string parse_role_condition(const Token& token,
                                               const std::string& source_label) {
    ExpressionParser parser(token.text, token.offset + 2, source_label);
    const std::string keyword = parser.read_word();
    if (keyword != "if" && keyword != "elif") {
        reject_template(source_label, token.offset + 2,
                        "an if or elif role condition", token.text);
    }
    const ReferenceKind reference = parser.parse_reference();
    if (reference != ReferenceKind::role) {
        reject_template(source_label, token.offset + 2,
                        "message role as the condition", token.text);
    }
    parser.expect_operator("==");
    const std::string role = parser.parse_string();
    parser.expect_end();
    if (role != "system" && role != "user" && role != "assistant") {
        reject_template(source_label, token.offset + 2,
                        "the role string system, user, or assistant", role);
    }
    return role;
}

[[nodiscard]] bool is_generation_condition(const Token& token,
                                           const std::string& source_label) {
    ExpressionParser parser(token.text, token.offset + 2, source_label);
    const std::string keyword = parser.read_word();
    if (keyword != "if") {
        return false;
    }
    const std::string first = parser.read_word();
    if (first != "loop") {
        return false;
    }
    parser.expect_character('.');
    parser.expect_word("last");
    parser.expect_word("and");
    parser.expect_word("add_generation_prompt");
    parser.expect_end();
    return true;
}

[[nodiscard]] std::vector<ExpressionTerm> parse_output(
        const Token& token, const std::string& source_label) {
    ExpressionParser parser(token.text, token.offset + 2, source_label);
    std::vector<ExpressionTerm> terms;
    while (true) {
        parser.skip_space();
        if (parser.at_end()) {
            reject_template(source_label, token.offset + 2,
                            "a non-empty output concatenation", "an empty expression");
        }
        const std::size_t before = parser.offset();
        const char first = token.text[before - (token.offset + 2)];
        if (first == '\'' || first == '"') {
            terms.push_back(ExpressionTerm{ExpressionTermKind::literal,
                                           parser.parse_string()});
        } else if (first == 'm') {
            const ReferenceKind reference = parser.parse_reference();
            if (reference != ReferenceKind::content) {
                reject_template(source_label, before,
                                "message content reference",
                                "message role or another field");
            }
            terms.push_back(ExpressionTerm{ExpressionTermKind::content, {}});
        } else {
            const std::string word = parser.read_word();
            if (word == "eos_token") {
                terms.push_back(ExpressionTerm{ExpressionTermKind::eos, {}});
            } else {
                reject_template(source_label, before,
                                "a literal, message content, or eos_token", word);
            }
        }
        parser.skip_space();
        if (parser.at_end()) {
            break;
        }
        parser.expect_character('+');
    }
    return terms;
}

[[nodiscard]] std::vector<ExpressionTerm> parse_generation_output(
        const Token& token, const std::string& source_label) {
    const std::vector<ExpressionTerm> terms = parse_output(token, source_label);
    for (const ExpressionTerm& term : terms) {
        if (term.kind != ExpressionTermKind::literal) {
            reject_template(source_label, token.offset + 2,
                            "generation output literals only",
                            "a message or tokenizer variable");
        }
    }
    return terms;
}

void append_checked(std::string& target, std::string_view value,
                    const std::string& source_label, std::size_t offset) {
    if (value.size() > target.max_size() - target.size()) {
        reject_template(source_label, offset, "a representable compiled literal",
                        "a literal-size overflow");
    }
    target.append(value);
}

[[nodiscard]] RoleAffix compile_role_branch(
        const std::vector<Token>& branch, const std::string& source_label,
        std::string_view eos_token) {
    RoleAffix result;
    bool output_seen = false;
    bool content_seen = false;
    bool eos_seen = false;
    for (const Token& token : branch) {
        if (token.kind == TokenKind::text) {
            require_whitespace_text(token, source_label);
            append_checked(output_seen ? result.after_content : result.before_content,
                           token.text, source_label, token.offset);
            continue;
        }
        if (token.kind != TokenKind::output || output_seen) {
            reject_template(source_label, token.offset,
                            "one role output concatenation and no nested blocks",
                            token.kind == TokenKind::block ? "a nested block"
                                                            : "multiple output tags");
        }
        output_seen = true;
        const std::vector<ExpressionTerm> terms = parse_output(token, source_label);
        for (const ExpressionTerm& term : terms) {
            if (term.kind == ExpressionTermKind::literal) {
                append_checked(content_seen ? result.after_content
                                             : result.before_content,
                               term.literal, source_label, token.offset);
            } else if (term.kind == ExpressionTermKind::content) {
                if (content_seen || eos_seen) {
                    reject_template(source_label, token.offset,
                                    "one message content reference before eos_token",
                                    "a repeated or late message content reference");
                }
                content_seen = true;
            } else {
                if (!content_seen || eos_seen) {
                    reject_template(source_label, token.offset,
                                    "one later eos_token after message content",
                                    "an eos_token in the wrong position");
                }
                eos_seen = true;
                append_checked(result.after_content, eos_token, source_label,
                               token.offset);
            }
        }
    }
    if (!output_seen) {
        reject_template(source_label, branch.empty() ? 0 : branch.front().offset,
                        "one role output concatenation", "no output tag");
    }
    if (!content_seen) {
        reject_template(source_label, branch.front().offset,
                        "exactly one message content reference",
                        "no message content reference");
    }
    if (!eos_seen) {
        reject_template(source_label, branch.front().offset,
                        "exactly one later eos_token", "no eos_token");
    }
    return result;
}

[[nodiscard]] std::string compile_generation_branch(
        const std::vector<Token>& branch, const std::string& source_label) {
    std::string result;
    bool output_seen = false;
    for (const Token& token : branch) {
        if (token.kind == TokenKind::text) {
            require_whitespace_text(token, source_label);
            append_checked(result, token.text, source_label, token.offset);
            continue;
        }
        if (token.kind != TokenKind::output || output_seen) {
            reject_template(source_label, token.offset,
                            "one generation literal concatenation and no nested blocks",
                            token.kind == TokenKind::block ? "a nested block"
                                                            : "multiple output tags");
        }
        output_seen = true;
        for (const ExpressionTerm& term :
             parse_generation_output(token, source_label)) {
            append_checked(result, term.literal, source_label, token.offset);
        }
    }
    if (!output_seen) {
        reject_template(source_label, branch.empty() ? 0 : branch.front().offset,
                        "one generation literal concatenation", "no output tag");
    }
    return result;
}

class TemplateCompiler {
public:
    TemplateCompiler(std::string source, std::string source_label,
                     std::string eos_token)
        : source_(std::move(source)), source_label_(std::move(source_label)),
          eos_token_(std::move(eos_token)),
          tokens_(tokenize_template(source_, source_label_)) {}

    [[nodiscard]] CompiledTemplate compile() {
        CompiledTemplate result;
        const std::size_t for_index = first_non_text(0);
        if (for_index == tokens_.size()) {
            reject_template(source_label_, 0, "one for message in messages block",
                            "no template block");
        }
        if (classify_block(tokens_[for_index]).kind != BlockKind::for_open) {
            reject_template(source_label_, tokens_[for_index].offset,
                            "the first block {% for message in messages %}",
                            tokens_[for_index].text);
        }
        append_text_range(result.prefix, 0, for_index);
        parse_for(tokens_[for_index], source_label_);
        std::size_t index = for_index + 1;

        const std::size_t role_if_index = first_non_text(index);
        if (role_if_index == tokens_.size()
            || classify_block(tokens_[role_if_index]).kind != BlockKind::if_open
            || is_generation_condition(tokens_[role_if_index], source_label_)) {
            reject_template(source_label_,
                            role_if_index == tokens_.size()
                                    ? source_.size()
                                    : tokens_[role_if_index].offset,
                            "one role-switch if block after the for block",
                            role_if_index == tokens_.size()
                                    ? "end of template"
                                    : tokens_[role_if_index].text);
        }
        append_text_range_to_roles(result, index, role_if_index);
        index = role_if_index;

        std::array<bool, 3> seen_roles{};
        std::size_t branch_count = 0;
        while (true) {
            if (index >= tokens_.size()
                || tokens_[index].kind != TokenKind::block) {
                reject_template(source_label_,
                                index == tokens_.size()
                                        ? source_.size()
                                        : tokens_[index].offset,
                                "a role branch closure", "end of template");
            }
            const BlockInfo info = classify_block(tokens_[index]);
            if (branch_count == 0 && info.kind != BlockKind::if_open) {
                reject_template(source_label_, tokens_[index].offset,
                                "the role-switch if block", tokens_[index].text);
            }
            if (branch_count != 0 && info.kind != BlockKind::elif) {
                reject_template(source_label_, tokens_[index].offset,
                                "exactly two role-switch elif blocks", tokens_[index].text);
            }
            if (branch_count >= 3) {
                reject_template(source_label_, tokens_[index].offset,
                                "exactly three role branches", "an extra role branch");
            }
            const std::string role = parse_role_condition(tokens_[index], source_label_);
            const std::size_t role_slot = role_index(role, source_label_, tokens_[index].offset);
            if (seen_roles[role_slot]) {
                reject_template(source_label_, tokens_[index].offset,
                                "each role exactly once", role);
            }
            seen_roles[role_slot] = true;
            ++branch_count;
            ++index;

            const std::size_t branch_end = next_block(index);
            std::vector<Token> branch(tokens_.begin() + index,
                                      tokens_.begin() + branch_end);
            RoleAffix affix = compile_role_branch(branch, source_label_, eos_token_);
            result.role_affixes[role_slot] = std::move(affix);
            index = branch_end;
            if (index == tokens_.size()) {
                reject_template(source_label_, source_.size(),
                                "endif closing the role switch", "end of template");
            }
            const BlockInfo end_info = classify_block(tokens_[index]);
            if (end_info.kind == BlockKind::elif) {
                if (branch_count >= 3) {
                    reject_template(source_label_, tokens_[index].offset,
                                    "exactly two elif role branches", "an extra elif");
                }
                continue;
            }
            if (end_info.kind == BlockKind::else_block) {
                reject_template(source_label_, tokens_[index].offset,
                                "no else branch in the role switch", "an else block");
            }
            if (end_info.kind != BlockKind::endif) {
                reject_template(source_label_, tokens_[index].offset,
                                "endif closing the role switch", tokens_[index].text);
            }
            parse_simple_block(tokens_[index], "endif", source_label_);
            ++index;
            break;
        }
        if (branch_count != 3 || !seen_roles[0] || !seen_roles[1] || !seen_roles[2]) {
            reject_template(source_label_, tokens_[role_if_index].offset,
                            "system, user, and assistant exactly once",
                            "an incomplete role switch");
        }
        const std::size_t possible_generation = first_non_text(index);
        if (possible_generation != tokens_.size()
            && classify_block(tokens_[possible_generation]).kind
                       == BlockKind::if_open
            && is_generation_condition(tokens_[possible_generation],
                                       source_label_)) {
            result.generation_inside_loop = true;
            append_text_range(result.iteration_tail, index, possible_generation);
            index = possible_generation + 1;
            const std::size_t generation_end = next_block(index);
            std::vector<Token> generation_branch(tokens_.begin() + index,
                                                 tokens_.begin() + generation_end);
            result.generation =
                    compile_generation_branch(generation_branch, source_label_);
            index = generation_end;
            if (index == tokens_.size()
                || classify_block(tokens_[index]).kind != BlockKind::endif) {
                reject_template(source_label_,
                                index == tokens_.size()
                                        ? source_.size()
                                        : tokens_[index].offset,
                                "endif closing the generation-prompt condition",
                                index == tokens_.size()
                                        ? "end of template"
                                        : tokens_[index].text);
            }
            parse_simple_block(tokens_[index], "endif", source_label_);
            ++index;
        }

        const std::size_t endfor_index = first_non_text(index);
        if (endfor_index == tokens_.size()
            || classify_block(tokens_[endfor_index]).kind != BlockKind::endfor) {
            reject_template(source_label_,
                            endfor_index == tokens_.size()
                                    ? source_.size()
                                    : tokens_[endfor_index].offset,
                            "endfor closing the message loop",
                            endfor_index == tokens_.size()
                                    ? "end of template"
                                    : tokens_[endfor_index].text);
        }
        parse_simple_block(tokens_[endfor_index], "endfor", source_label_);
        if (result.generation_inside_loop) {
            append_text_range(result.iteration_tail_after_generation, index,
                              endfor_index);
        } else {
            append_text_range(result.iteration_tail, index, endfor_index);
        }
        index = endfor_index + 1;

        if (result.generation_inside_loop) {
            append_text_range(result.suffix, index, tokens_.size());
            prepend_loop_prefix(result, loop_prefix_);
            return result;
        }
        const std::size_t after_loop_end = first_non_text(index);
        if (after_loop_end == tokens_.size()) {
            append_text_range(result.suffix, index, tokens_.size());
            prepend_loop_prefix(result, loop_prefix_);
            return result;
        }
        if (classify_block(tokens_[after_loop_end]).kind != BlockKind::if_open
            || !is_generation_condition(tokens_[after_loop_end], source_label_)) {
            reject_template(source_label_, tokens_[after_loop_end].offset,
                            "an optional final generation-prompt if block",
                            tokens_[after_loop_end].text);
        }
        append_text_range(result.post_loop, index, after_loop_end);
        index = after_loop_end + 1;
        const std::size_t generation_end = next_block(index);
        std::vector<Token> generation_branch(tokens_.begin() + index,
                                             tokens_.begin() + generation_end);
        result.generation = compile_generation_branch(generation_branch, source_label_);
        index = generation_end;
        if (index == tokens_.size()
            || classify_block(tokens_[index]).kind != BlockKind::endif) {
            reject_template(source_label_,
                            index == tokens_.size()
                                    ? source_.size()
                                    : tokens_[index].offset,
                            "endif closing the generation-prompt condition",
                            index == tokens_.size()
                                    ? "end of template"
                                    : tokens_[index].text);
        }
        parse_simple_block(tokens_[index], "endif", source_label_);
        ++index;
        append_text_range(result.suffix, index, tokens_.size());
        prepend_loop_prefix(result, loop_prefix_);
        return result;
    }

private:
    [[nodiscard]] std::size_t first_non_text(std::size_t begin) const {
        std::size_t index = begin;
        while (index < tokens_.size()
               && tokens_[index].kind == TokenKind::text) {
            ++index;
        }
        return index;
    }

    [[nodiscard]] std::size_t next_block(std::size_t begin) const {
        std::size_t index = begin;
        while (index < tokens_.size()
               && tokens_[index].kind != TokenKind::block) {
            ++index;
        }
        return index;
    }

    void append_text_range(std::string& destination, std::size_t begin,
                           std::size_t end) const {
        for (std::size_t index = begin; index < end; ++index) {
            if (tokens_[index].kind != TokenKind::text) {
                reject_template(source_label_, tokens_[index].offset,
                                "whitespace literal text only", tokens_[index].text);
            }
            require_whitespace_text(tokens_[index], source_label_);
            append_checked(destination, tokens_[index].text, source_label_,
                           tokens_[index].offset);
        }
    }

    void append_text_range_to_roles(CompiledTemplate& result, std::size_t begin,
                                    std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            if (tokens_[index].kind != TokenKind::text) {
                reject_template(source_label_, tokens_[index].offset,
                                "whitespace before the role switch", tokens_[index].text);
            }
            require_whitespace_text(tokens_[index], source_label_);
            append_checked(loop_prefix_, tokens_[index].text, source_label_,
                           tokens_[index].offset);
        }
        (void)result;
    }

    void prepend_loop_prefix(CompiledTemplate& result,
                             const std::string& loop_prefix) const {
        for (RoleAffix& affix : result.role_affixes) {
            std::string combined;
            combined.reserve(loop_prefix.size() + affix.before_content.size());
            combined.append(loop_prefix);
            combined.append(affix.before_content);
            affix.before_content = std::move(combined);
        }
    }

    std::string source_;
    std::string source_label_;
    std::string eos_token_;
    std::vector<Token> tokens_;
    std::string loop_prefix_;
};

[[nodiscard]] CompiledTemplate compile_template(
        std::string source, const std::string& source_label,
        std::string eos_token) {
    validate_utf8_and_line_endings(source, source_label);
    return TemplateCompiler(std::move(source), source_label, std::move(eos_token))
            .compile();
}

}  // namespace

struct ChatFormatter::Impl {
    std::string selected_template;
    std::array<RoleAffix, 3> role_affixes;
    std::string generation;
    std::string iteration_tail;
    std::string iteration_tail_after_generation;
    bool generation_inside_loop = false;
    std::string prefix;
    std::string post_loop;
    std::string suffix;
};

ChatFormatter::ChatFormatter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ChatFormatter::~ChatFormatter() = default;

std::string ChatFormatter::format(
        std::span<const ChatMessageView> messages,
        bool add_generation_prompt) const {
    std::string output;
    std::size_t output_size = 0;
    const auto add_size = [&](std::size_t size) {
        if (size > output.max_size() - output_size) {
            throw std::overflow_error("chat format output size overflows");
        }
        output_size += size;
    };
    add_size(impl_->prefix.size());
    add_size(impl_->post_loop.size());
    add_size(impl_->suffix.size());
    if (add_generation_prompt && !impl_->generation_inside_loop) {
        add_size(impl_->generation.size());
    }
    for (std::size_t message_index = 0; message_index < messages.size();
         ++message_index) {
        const ChatMessageView& message = messages[message_index];
        const std::size_t index =
                message.role == "system"
                        ? 0
                        : message.role == "user" ? 1
                                                 : message.role == "assistant" ? 2
                                                                                : 3;
        if (index == 3) {
            throw std::invalid_argument(
                    "chat message role must be system, user, or assistant; actual "
                    + std::string(message.role));
        }
        const RoleAffix& affix = impl_->role_affixes[index];
        add_size(affix.before_content.size());
        add_size(message.content.size());
        add_size(affix.after_content.size());
        add_size(impl_->iteration_tail.size());
        if (impl_->generation_inside_loop && add_generation_prompt
            && message_index + 1 == messages.size()) {
            add_size(impl_->generation.size());
        }
        add_size(impl_->iteration_tail_after_generation.size());
    }
    output.reserve(output_size);
    output.append(impl_->prefix);
    for (std::size_t message_index = 0; message_index < messages.size();
         ++message_index) {
        const ChatMessageView& message = messages[message_index];
        const std::size_t index =
                message.role == "system"
                        ? 0
                        : message.role == "user" ? 1
                                                 : 2;
        const RoleAffix& affix = impl_->role_affixes[index];
        output.append(affix.before_content);
        output.append(message.content);
        output.append(affix.after_content);
        output.append(impl_->iteration_tail);
        if (impl_->generation_inside_loop && add_generation_prompt
            && message_index + 1 == messages.size()) {
            output.append(impl_->generation);
        }
        output.append(impl_->iteration_tail_after_generation);
    }
    output.append(impl_->post_loop);
    if (add_generation_prompt && !impl_->generation_inside_loop) {
        output.append(impl_->generation);
    }
    output.append(impl_->suffix);
    return output;
}

std::unique_ptr<ChatFormatter> load_chat_formatter(
        const std::filesystem::path& model_directory,
        std::optional<std::string_view> template_override) {
    const std::filesystem::path config_path =
            model_directory / kTokenizerConfigFileName;
    const std::string config_text = read_file(config_path);
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(config_text);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(
                "invalid tokenizer configuration: " + config_path.string()
                + " field '<tokenizer_config.json>' requires a valid JSON document; actual "
                + error.what());
    }

    // An engaged override has precedence even if the distribution's selected
    // chat_template value is not a string. All tokenizer policy metadata is
    // still validated before the override is compiled.
    validate_metadata(document, config_path, !template_override.has_value());
    std::string source = selected_template(document, config_path, template_override);
    const std::string source_label = template_override.has_value()
                                             ? "template_override"
                                             : config_path.string() + " field 'chat_template'";
    const auto& eos = document.at("eos_token").get_ref<const std::string&>();
    CompiledTemplate compiled =
            compile_template(source, source_label, std::string(eos));

    auto impl = std::make_unique<ChatFormatter::Impl>();
    impl->selected_template = std::move(source);
    impl->role_affixes = std::move(compiled.role_affixes);
    impl->generation = std::move(compiled.generation);
    impl->iteration_tail = std::move(compiled.iteration_tail);
    impl->iteration_tail_after_generation =
            std::move(compiled.iteration_tail_after_generation);
    impl->generation_inside_loop = compiled.generation_inside_loop;
    impl->prefix = std::move(compiled.prefix);
    impl->post_loop = std::move(compiled.post_loop);
    impl->suffix = std::move(compiled.suffix);
    return std::unique_ptr<ChatFormatter>(new ChatFormatter(std::move(impl)));
}

}  // namespace iom
