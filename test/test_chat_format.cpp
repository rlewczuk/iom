#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "iom/chat_format.hpp"

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
