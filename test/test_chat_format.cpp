#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/chat_format.hpp"
#include "iom/tokenizer.hpp"
#include "tokenizer_fixture.hpp"

namespace {

class TempDir {
public:
    explicit TempDir(std::string tag)
        : path_(std::filesystem::temp_directory_path()
                / ("iom-chat-format-" + std::move(tag))) {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (!std::filesystem::create_directories(path_)) {
            throw std::runtime_error("cannot create test directory: "
                                     + path_.string());
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

std::string compact_template() {
    return R"({% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %})";
}

std::string sentence_template() {
    return "{% for message in messages %}"
           "{% if message['role'] == 'system' %}"
           "{{ '<|system|>\\n' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'user' %}"
           "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'assistant' %}"
           "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n"
           "{% endif %}"
           "{% if loop.last and add_generation_prompt %}"
           "{{ '<|assistant|>\\n' }}"
           "{% endif %}"
           "{% endfor %}";
}


nlohmann::json token_metadata(std::string content, bool special = true) {
    return nlohmann::json{{"content", std::move(content)},
                          {"lstrip", false},
                          {"normalized", false},
                          {"rstrip", false},
                          {"single_word", false},
                          {"special", special}};
}

nlohmann::json valid_config(std::string chat_template = compact_template()) {
    return nlohmann::json{
            {"added_tokens_decoder",
             nlohmann::json{{"0", token_metadata("<unk>")},
                            {"1", token_metadata("<s>")},
                            {"2", token_metadata("</s>")}}},
            {"bos_token", "<s>"},
            {"chat_template", std::move(chat_template)},
            {"clean_up_tokenization_spaces", false},
            {"eos_token", "</s>"},
            {"legacy", false},
            {"model_max_length", 2048},
            {"pad_token", "</s>"},
            {"padding_side", "right"},
            {"sp_model_kwargs", nlohmann::json::object()},
            {"tokenizer_class", "LlamaTokenizer"},
            {"unk_token", "<unk>"},
            {"use_default_system_prompt", false},
    };
}

void write_config(const TempDir& directory, const nlohmann::json& document) {
    std::ofstream output(directory.path() / "tokenizer_config.json",
                         std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output << document.dump();
    REQUIRE(output.good());
}

std::string invalid_reason(const TempDir& directory,
                           std::optional<std::string_view> override_template =
                                   std::nullopt) {
    try {
        auto formatter = iom::load_chat_formatter(directory.path(),
                                                   override_template);
        (void)formatter;
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

std::string format_invalid_reason(
        const iom::ChatFormatter& formatter,
        std::span<const iom::ChatMessageView> messages) {
    try {
        (void)formatter.format(messages, false);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}


}  // namespace

using iom_tokenizer_test::write_tokenizer;

std::string distribution_template() {
    return "{% for message in messages %}\n"
           "{% if message['role'] == 'user' %}\n"
           "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'system' %}\n"
           "{{ '<|system|>\\n' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'assistant' %}\n"
           "{{ '<|assistant|>\\n' + message['content'] + eos_token }}\n"
           "{% endif %}\n"
           "{% if loop.last and add_generation_prompt %}\n"
           "{{ '<|assistant|>' }}\n"
           "{% endif %}\n"
           "{% endfor %}";
}

std::string fixed_override_template() {
    return "{% for message in messages %}"
           "{% if message['role'] == 'system' %}"
           "{{ 'OVERRIDE_SYSTEM:' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'user' %}"
           "{{ 'OVERRIDE_USER:' + message['content'] + eos_token }}\n"
           "{% elif message['role'] == 'assistant' %}"
           "{{ 'OVERRIDE_ASSISTANT:' + message['content'] + eos_token }}\n"
           "{% endif %}"
           "{% if loop.last and add_generation_prompt %}"
           "{{ 'OVERRIDE_GENERATION:' }}"
           "{% endif %}"
           "{% endfor %}";
}

void write_composition_artifact(const TempDir& directory) {
    write_tokenizer(directory.path());
    write_config(directory, valid_config(distribution_template()));
}

void check_prompt_encoding(
        const iom::Tokenizer& tokenizer, std::string_view rendered,
        const std::vector<std::uint32_t>& expected_without_specials,
        const std::vector<std::uint32_t>& expected_with_specials,
        std::string_view decoded_without_specials,
        std::string_view decoded_skip_specials,
        std::string_view decoded_with_specials) {
    const auto without_specials =
            tokenizer.encode(rendered, iom::EncodeOptions{false});
    CHECK(without_specials == expected_without_specials);
    CHECK(tokenizer.decode(without_specials, iom::DecodeOptions{false})
          == decoded_without_specials);
    CHECK(tokenizer.decode(without_specials, iom::DecodeOptions{true})
          == decoded_skip_specials);

    const auto with_specials =
            tokenizer.encode(rendered, iom::EncodeOptions{true});
    CHECK(with_specials == expected_with_specials);
    CHECK(tokenizer.decode(with_specials, iom::DecodeOptions{false})
          == decoded_with_specials);
    CHECK(tokenizer.decode(with_specials, iom::DecodeOptions{true})
          == decoded_skip_specials);
}

TEST_CASE("Chat format loads the validated default and formats owned literals") {
    const TempDir directory("default");
    write_config(directory, valid_config());

    const auto formatter = iom::load_chat_formatter(directory.path());
    const iom::ChatMessageView messages[] = {{"system", "rules"},
                                             {"user", "hello"},
                                             {"assistant", "answer"}};
    CHECK(formatter != nullptr);
    CHECK(formatter->format(messages, false)
          == "<|system|>\nrules</s><|user|>\nhello</s><|assistant|>\nanswer</s>");
    CHECK(formatter->format(std::span<const iom::ChatMessageView>(messages, 2), true)
          == "<|system|>\nrules</s><|user|>\nhello</s><|assistant|>\n");
    CHECK(formatter->format(std::span<const iom::ChatMessageView>(), true) == "");
}

TEST_CASE("Chat format accepts quote and indexing variants and retains owner bytes") {
    const TempDir directory("override-variants");
    nlohmann::json document = valid_config("not compiled");
    write_config(directory, document);
    const std::string override_template =
            R"({% for message in messages %}{% if message.role == "assistant" %}{{ "A:" + message["content"] + eos_token }}{% elif message["role"] == "system" %}{{ "S:" + message.content + eos_token }}{% elif message['role'] == 'user' %}{{ 'U:' + message['content'] + eos_token }}{% endif %}{% endfor %})";

    const auto formatter =
            iom::load_chat_formatter(directory.path(), override_template);
    std::error_code error;
    std::filesystem::remove_all(directory.path(), error);
    const iom::ChatMessageView messages[] = {{"assistant", "ok"},
                                             {"system", "rules"},
                                             {"user", "hello"}};
    CHECK(formatter->format(messages, false) == "A:ok</s>S:rules</s>U:hello</s>");
}

TEST_CASE("Chat format gives an engaged override precedence without default fallback") {
    const TempDir directory("override-precedence");
    nlohmann::json document = valid_config();
    document["chat_template"] = 7;
    write_config(directory, document);
    const std::string override_template =
            R"({% for message in messages %}{% if message['role'] == 'system' %}{{ 'S' + message['content'] + eos_token }}{% elif message['role'] == 'user' %}{{ 'U' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ 'A' + message['content'] + eos_token }}{% endif %}{% endfor %})";

    const auto formatter =
            iom::load_chat_formatter(directory.path(), override_template);
    REQUIRE(formatter != nullptr);
    CHECK(formatter->format(std::span<const iom::ChatMessageView>(), false)
          == "");
    CHECK(!invalid_reason(directory, std::string_view("")).empty());
}

TEST_CASE("Chat format rejects schema policy and malformed JSON with context") {
    const TempDir directory("schema");
    nlohmann::json document = valid_config();
    document["added_tokens_decoder"]["1"]["special"] = false;
    write_config(directory, document);
    CHECK(invalid_reason(directory).find("field 'special'")
          != std::string::npos);

    write_config(directory, nlohmann::json{{"chat_template", "broken"}});
    const std::string missing = invalid_reason(directory);
    CHECK(missing.find("added_tokens_decoder") != std::string::npos);

    std::ofstream malformed(directory.path() / "tokenizer_config.json",
                            std::ios::binary | std::ios::trunc);
    malformed << "{";
    malformed.close();
    CHECK(invalid_reason(directory).find("valid JSON document")
          != std::string::npos);
}

TEST_CASE("Chat format rejects empty overrides and restricted grammar violations") {
    const TempDir directory("grammar");
    write_config(directory, valid_config());
    CHECK(invalid_reason(directory, std::string_view(""))
          .find("byte offset") != std::string::npos);

    const std::string invalid_templates[] = {
            R"({% for message in messages %}{% if message['role'] == 'system' %}{{ 'S' + message['content'] + eos_token }}{% else %}{{ 'U' + message['content'] + eos_token }}{% endif %}{% endfor %})",
            R"({% for message in messages %}{% if message['role'] == 'system' %}{% if message['role'] == 'user' %}{{ 'S' + message['content'] + eos_token }}{% endif %}{% elif message['role'] == 'user' %}{{ 'U' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ 'A' + message['content'] + eos_token }}{% endif %}{% endfor %})",
            R"({% for message in messages %}{% if message['role'] == 'system' %}{{ 'S' + message['content']|trim + eos_token }}{% elif message['role'] == 'user' %}{{ 'U' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ 'A' + message['content'] + eos_token }}{% endif %}{% endfor %})",
            R"({% for message in messages %}{% if message['role'] == 'system' %}{{ 'S' + message['content'] + eos_token }}{% elif message['role'] == 'user' %}{{ 'U' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ 'A' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'A' }}{% endif %})",
    };
    for (const std::string& source : invalid_templates) {
        CHECK(invalid_reason(directory, source).find("invalid chat template")
              != std::string::npos);
    }
}

TEST_CASE("Chat format validates escapes UTF-8 line endings and whitespace lowering") {
    const TempDir directory("whitespace");
    write_config(directory, valid_config());
    const std::string source =
            "\n{% for message in messages %}\n\t{% if message['role'] == 'user' %}\n  {{ 'U\\n' + message['content'] + eos_token }}\n\t{% elif message['role'] == 'system' %}\n  {{ 'S\\t' + message['content'] + eos_token }}\n\t{% elif message['role'] == 'assistant' %}\n  {{ 'A\\r' + message['content'] + eos_token }}\n\t{% endif %}\n{% endfor %}\n{% if loop.last and add_generation_prompt %}\n{{ 'G' }}\n{% endif %}\n";
    const auto formatter = iom::load_chat_formatter(directory.path(), source);
    const iom::ChatMessageView message[] = {{"user", "x"}};
    CHECK(formatter->format(message, false) == "\n  U\nx</s>\n");
    CHECK(formatter->format(message, true) == "\n  U\nx</s>\nG\n");

    const std::string invalid_escape =
            R"({% for message in messages %}{% if message['role'] == 'system' %}{{ '\q' + message['content'] + eos_token }}{% elif message['role'] == 'user' %}{{ 'U' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ 'A' + message['content'] + eos_token }}{% endif %}{% endfor %})";
    CHECK(invalid_reason(directory, invalid_escape).find("supported escape")
          != std::string::npos);
    const std::string raw_cr = compact_template() + "\r";
    CHECK(invalid_reason(directory, raw_cr).find("raw CR") != std::string::npos);
    const std::string invalid_utf8 = compact_template() + std::string("\xC3\x28", 2);
    CHECK(invalid_reason(directory, invalid_utf8).find("valid UTF-8")
          != std::string::npos);
}

TEST_CASE("Chat format rejects unknown render roles") {
    const TempDir directory("unknown-role");
    write_config(directory, valid_config());
    const auto formatter = iom::load_chat_formatter(directory.path());
    const iom::ChatMessageView message[] = {{"tool", "ignored"}};
    CHECK_THROWS_AS(formatter->format(message, false), std::invalid_argument);
}

TEST_CASE("Chat format renders sentence-style messages with exact line feeds") {
    const TempDir directory("render-sentences");
    write_config(directory, valid_config(sentence_template()));
    const auto formatter = iom::load_chat_formatter(directory.path());

    const iom::ChatMessageView one_user[] = {{"user", "x"}};
    CHECK(formatter->format(one_user, false) == "<|user|>\nx</s>\n");
    CHECK(formatter->format(one_user, true)
          == "<|user|>\nx</s>\n<|assistant|>\n");

    const iom::ChatMessageView messages[] = {{"system", "rules"},
                                             {"user", "hello"},
                                             {"assistant", "answer"}};
    CHECK(formatter->format(messages, false)
          == "<|system|>\nrules</s>\n"
             "<|user|>\nhello</s>\n"
             "<|assistant|>\nanswer</s>\n");

    const iom::ChatMessageView empty_user[] = {{"user", ""}};
    CHECK(formatter->format(empty_user, false) == "<|user|>\n</s>\n");
    CHECK(formatter->format(empty_user, true)
          == "<|user|>\n</s>\n<|assistant|>\n");
    CHECK(formatter->format(one_user, false) == "<|user|>\nx</s>\n");
}

TEST_CASE("Chat format renders a valid override with independent owner state") {
    const TempDir directory("render-override");
    const std::string override_template =
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ 'S:' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'user' %}"
            "{{ 'U:' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'A:' + message['content'] + eos_token }}\n"
            "{% endif %}"
            "{% if loop.last and add_generation_prompt %}"
            "{{ 'A:\\n' }}"
            "{% endif %}"
            "{% endfor %}";
    write_config(directory, valid_config("not compiled"));
    const auto formatter =
            iom::load_chat_formatter(directory.path(), override_template);
    const iom::ChatMessageView message[] = {{"user", "hello"}};
    CHECK(formatter->format(message, false) == "U:hello</s>\n");
    CHECK(formatter->format(message, true) == "U:hello</s>\nA:\n");
    CHECK(formatter->format(message, false) == "U:hello</s>\n");
}

TEST_CASE("Chat format rejects invalid message input before rendering") {
    const TempDir directory("render-validation");
    write_config(directory, valid_config(sentence_template()));
    const auto formatter = iom::load_chat_formatter(directory.path());

    const iom::ChatMessageView invalid_role[] = {{"User", "x"}};
    const std::string role_error =
            format_invalid_reason(*formatter, invalid_role);
    CHECK(role_error.find("field 'role'") != std::string::npos);
    CHECK(role_error.find("User") != std::string::npos);

    const std::string invalid_bytes("\xC3\x28", 2);
    const iom::ChatMessageView invalid_content[] = {{"user", invalid_bytes}};
    const std::string content_error =
            format_invalid_reason(*formatter, invalid_content);
    CHECK(content_error.find("field 'content'") != std::string::npos);
    CHECK(content_error.find("valid UTF-8") != std::string::npos);

    const iom::ChatMessageView valid_message[] = {{"user", "x"}};
    CHECK(formatter->format(valid_message, false) == "<|user|>\nx</s>\n");
}

TEST_CASE("Chat format composes raw and structured owner paths") {
    const TempDir directory("composition");
    write_composition_artifact(directory);

    const auto tokenizer = iom::load_tokenizer(directory.path());
    const auto formatter = iom::load_chat_formatter(directory.path());
    REQUIRE(tokenizer != nullptr);
    REQUIRE(formatter != nullptr);

    // Raw text bypasses the formatter and retains the tokenizer's own BOS
    // policy.
    CHECK(tokenizer->encode("hello", iom::EncodeOptions{false})
          == std::vector<std::uint32_t>{268});
    CHECK(tokenizer->encode("hello", iom::EncodeOptions{true})
          == std::vector<std::uint32_t>{1, 268});
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{268}, {})
          == "hello");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1, 268}, {})
          == "<s> hello");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1, 268},
                            iom::DecodeOptions{true})
          == "hello");

    CHECK(tokenizer->encode("<s>", iom::EncodeOptions{false})
          == std::vector<std::uint32_t>{1});
    CHECK(tokenizer->encode("<s>", iom::EncodeOptions{true})
          == std::vector<std::uint32_t>{1, 1});
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1}, {}) == "<s>");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1, 1}, {})
          == "<s><s>");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1, 1},
                            iom::DecodeOptions{true})
          == "");

    const std::span<const iom::ChatMessageView> empty_messages;
    CHECK(formatter->format(empty_messages, false) == "");
    CHECK(formatter->format(empty_messages, true) == "");
    CHECK(tokenizer->encode("", iom::EncodeOptions{false})
          == std::vector<std::uint32_t>{});
    CHECK(tokenizer->encode("", iom::EncodeOptions{true})
          == std::vector<std::uint32_t>{1});
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{}, {}) == "");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1}, {}) == "<s>");
    CHECK(tokenizer->decode(std::vector<std::uint32_t>{1},
                            iom::DecodeOptions{true})
          == "");

    const iom::ChatMessageView system_user[] = {
            {"system", "system"},
            {"user", "hello"},
    };
    const std::string system_user_rendered =
            formatter->format(system_user, false);
    CHECK(system_user_rendered
          == "<|system|>\nsystem</s>\n<|user|>\nhello</s>\n");
    check_prompt_encoding(
            *tokenizer, system_user_rendered,
            std::vector<std::uint32_t>{
                    259, 63, 127, 118, 124, 118, 119, 261, 112, 127, 65,
                    13,  118, 124, 118, 119, 261, 112, 2,   259, 13,  63,
                    127, 120, 118, 261, 270, 127, 65,  13,  267, 2,   259,
                    13},
            std::vector<std::uint32_t>{
                    1,   259, 63, 127, 118, 124, 118, 119, 261, 112, 127,
                    65,  13,  118, 124, 118, 119, 261, 112, 2, 259, 13,
                    63,  127, 120, 118, 261, 270, 127, 65, 13, 267, 2,
                    259, 13},
            "<|system|>\nsystem</s> \n<|user|>\nhello</s> \n",
            "<|system|>\nsystem \n<|user|>\nhello \n",
            "<s> <|system|>\nsystem</s> \n<|user|>\nhello</s> \n");

    const std::string system_user_generation =
            formatter->format(system_user, true);
    CHECK(system_user_generation
          == "<|system|>\nsystem</s>\n"
             "<|user|>\nhello</s>\n"
             "<|assistant|>\n");
    check_prompt_encoding(
            *tokenizer, system_user_generation,
            std::vector<std::uint32_t>{
                    259, 63, 127, 118, 124, 118, 119, 261, 112, 127, 65,
                    13,  118, 124, 118, 119, 261, 112, 2,   259, 13,  63,
                    127, 120, 118, 261, 270, 127, 65,  13,  267, 2,   259,
                    13,  63,  127, 100, 118, 118, 108, 118, 119, 100, 113,
                    119, 127, 65, 13},
            std::vector<std::uint32_t>{
                    1,   259, 63, 127, 118, 124, 118, 119, 261, 112, 127,
                    65,  13,  118, 124, 118, 119, 261, 112, 2, 259, 13,
                    63,  127, 120, 118, 261, 270, 127, 65, 13, 267, 2,
                    259, 13, 63, 127, 100, 118, 118, 108, 118, 119, 100,
                    113, 119, 127, 65, 13},
            "<|system|>\nsystem</s> \n<|user|>\nhello</s> \n"
            "<|assistant|>\n",
            "<|system|>\nsystem \n<|user|>\nhello \n"
            "<|assistant|>\n",
            "<s> <|system|>\nsystem</s> \n<|user|>\nhello</s> \n"
            "<|assistant|>\n");

    const iom::ChatMessageView assistant_messages[] = {
            {"user", "hello"},
            {"assistant", "world"},
    };
    const std::string assistant_rendered =
            formatter->format(assistant_messages, false);
    CHECK(assistant_rendered
          == "<|user|>\nhello</s>\n"
             "<|assistant|>\nworld</s>\n");
    check_prompt_encoding(
            *tokenizer, assistant_rendered,
            std::vector<std::uint32_t>{
                    259, 63, 127, 120, 118, 261, 270, 127, 65,  13,  267,
                    2,   259, 13,  63, 127, 100, 118, 118, 108, 118, 119,
                    100, 113, 119, 127, 65, 13, 275, 2, 259, 13},
            std::vector<std::uint32_t>{
                    1,   259, 63, 127, 120, 118, 261, 270, 127, 65, 13,
                    267, 2,   259, 13, 63, 127, 100, 118, 118, 108, 118,
                    119, 100, 113, 119, 127, 65, 13, 275, 2, 259, 13},
            "<|user|>\nhello</s> \n<|assistant|>\nworld</s> \n",
            "<|user|>\nhello \n<|assistant|>\nworld \n",
            "<s> <|user|>\nhello</s> \n<|assistant|>\nworld</s> \n");

    const iom::ChatMessageView literal_message[] = {{"user", "<s>"}};
    const std::string literal_rendered =
            formatter->format(literal_message, false);
    CHECK(literal_rendered == "<|user|>\n<s></s>\n");
    check_prompt_encoding(
            *tokenizer, literal_rendered,
            std::vector<std::uint32_t>{
                    259, 63, 127, 120, 118, 261, 270, 127, 65, 13, 1, 2,
                    259, 13},
            std::vector<std::uint32_t>{
                    1, 259, 63, 127, 120, 118, 261, 270, 127, 65, 13, 1,
                    2, 259, 13},
            "<|user|>\n<s></s> \n",
            "<|user|>\n \n",
            "<s> <|user|>\n<s></s> \n");
}

TEST_CASE("Chat format composes formatter override and generation policy") {
    const TempDir directory("composition-override");
    write_composition_artifact(directory);

    const auto tokenizer = iom::load_tokenizer(directory.path());
    const auto default_formatter = iom::load_chat_formatter(directory.path());
    const auto override_formatter =
            iom::load_chat_formatter(directory.path(), fixed_override_template());
    REQUIRE(tokenizer != nullptr);
    REQUIRE(default_formatter != nullptr);
    REQUIRE(override_formatter != nullptr);

    const iom::ChatMessageView message[] = {{"user", "hello"}};
    const std::string default_rendered =
            default_formatter->format(message, true);
    const std::string override_rendered =
            override_formatter->format(message, false);
    const std::string override_generation =
            override_formatter->format(message, true);
    CHECK(default_rendered
          == "<|user|>\nhello</s>\n<|assistant|>\n");
    CHECK(override_rendered == "OVERRIDE_USER:hello</s>\n");
    CHECK(override_generation
          == "OVERRIDE_USER:hello</s>\nOVERRIDE_GENERATION:");
    CHECK(default_rendered != override_generation);
    CHECK(override_rendered.find("</s>") != std::string::npos);
    CHECK(override_generation.find("</s>") != std::string::npos);

    check_prompt_encoding(
            *tokenizer, override_rendered,
            std::vector<std::uint32_t>{
                    259, 82, 89, 72, 85, 85, 76, 71, 72, 98, 88, 86, 72,
                    85, 61, 267, 2, 259, 13},
            std::vector<std::uint32_t>{
                    1, 259, 82, 89, 72, 85, 85, 76, 71, 72, 98, 88, 86, 72,
                    85, 61, 267, 2, 259, 13},
            "OVERRIDE_USER:hello</s> \n", "OVERRIDE_USER:hello \n",
            "<s> OVERRIDE_USER:hello</s> \n");
    check_prompt_encoding(
            *tokenizer, override_generation,
            std::vector<std::uint32_t>{
                    259, 82, 89, 72, 85, 85, 76, 71, 72, 98, 88, 86, 72,
                    85, 61, 267, 2, 259, 13, 82, 89, 72, 85, 85, 76, 71,
                    72, 98, 74, 72, 81, 72, 85, 68, 87, 76, 82, 81, 61},
            std::vector<std::uint32_t>{
                    1, 259, 82, 89, 72, 85, 85, 76, 71, 72, 98, 88, 86,
                    72, 85, 61, 267, 2, 259, 13, 82, 89, 72, 85, 85, 76,
                    71, 72, 98, 74, 72, 81, 72, 85, 68, 87, 76, 82, 81,
                    61},
            "OVERRIDE_USER:hello</s> \nOVERRIDE_GENERATION:",
            "OVERRIDE_USER:hello \nOVERRIDE_GENERATION:",
            "<s> OVERRIDE_USER:hello</s> \nOVERRIDE_GENERATION:");

    const std::string literal_override =
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ 'S:' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<s>' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'A:' + message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}";
    const auto literal_formatter =
            iom::load_chat_formatter(directory.path(), literal_override);
    REQUIRE(literal_formatter != nullptr);
    const std::string literal_rendered =
            literal_formatter->format(message, false);
    CHECK(literal_rendered == "<s>hello</s>");
    check_prompt_encoding(
            *tokenizer, literal_rendered,
            std::vector<std::uint32_t>{1, 268, 2},
            std::vector<std::uint32_t>{1, 1, 268, 2},
            "<s> hello</s>", "hello", "<s><s> hello</s>");
}

TEST_CASE("Chat format retains contextual boundary failures atomically") {
    const TempDir directory("composition-errors");
    write_composition_artifact(directory);
    const auto tokenizer = iom::load_tokenizer(directory.path());
    const auto formatter = iom::load_chat_formatter(directory.path());
    REQUIRE(tokenizer != nullptr);
    REQUIRE(formatter != nullptr);

    std::unique_ptr<iom::ChatFormatter> malformed_formatter;
    std::string malformed_reason;
    try {
        malformed_formatter = iom::load_chat_formatter(
                directory.path(), std::string_view("{% for"));
    } catch (const std::invalid_argument& error) {
        malformed_reason = error.what();
    }
    CHECK(malformed_formatter == nullptr);
    CHECK(malformed_reason.find("template_override") != std::string::npos);
    CHECK(malformed_reason.find("byte offset") != std::string::npos);

    const iom::ChatMessageView invalid_role[] = {{"User", "x"}};
    std::string published_format = "sentinel";
    std::string role_reason;
    try {
        published_format = formatter->format(invalid_role, false);
    } catch (const std::invalid_argument& error) {
        role_reason = error.what();
    }
    CHECK(published_format == "sentinel");
    CHECK(role_reason.find("field 'role'") != std::string::npos);
    CHECK(role_reason.find("User") != std::string::npos);

    const std::string invalid_bytes("\xC3\x28", 2);
    const iom::ChatMessageView invalid_content[] = {
            {"user", invalid_bytes}};
    published_format = "sentinel";
    std::string content_reason;
    try {
        published_format = formatter->format(invalid_content, false);
    } catch (const std::invalid_argument& error) {
        content_reason = error.what();
    }
    CHECK(published_format == "sentinel");
    CHECK(content_reason.find("field 'content'") != std::string::npos);
    CHECK(content_reason.find("valid UTF-8") != std::string::npos);

    const std::array<std::uint32_t, 3> invalid_ids = {
            267, std::numeric_limits<std::uint32_t>::max(), 267};
    std::string published_decode = "sentinel";
    std::string decode_reason;
    try {
        published_decode = tokenizer->decode(invalid_ids, {});
    } catch (const std::invalid_argument& error) {
        decode_reason = error.what();
    }
    CHECK(published_decode == "sentinel");
    CHECK(decode_reason.find("index 1") != std::string::npos);
    CHECK(decode_reason.find(
                  std::to_string(std::numeric_limits<std::uint32_t>::max()))
          != std::string::npos);
    CHECK(decode_reason.find("0..31999") != std::string::npos);
}
