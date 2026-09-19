#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace iom {

struct ChatMessageView {
    std::string_view role;
    std::string_view content;
};

class ChatFormatter final {
public:
    ~ChatFormatter();

    ChatFormatter(const ChatFormatter&) = delete;
    ChatFormatter& operator=(const ChatFormatter&) = delete;
    ChatFormatter(ChatFormatter&&) = delete;
    ChatFormatter& operator=(ChatFormatter&&) = delete;

    [[nodiscard]] std::string format(
            std::span<const ChatMessageView> messages,
            bool add_generation_prompt) const;

private:
    struct Impl;

    explicit ChatFormatter(std::unique_ptr<Impl> impl);

    friend std::unique_ptr<ChatFormatter> load_chat_formatter(
            const std::filesystem::path&,
            std::optional<std::string_view>);

    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::unique_ptr<ChatFormatter> load_chat_formatter(
        const std::filesystem::path& model_directory,
        std::optional<std::string_view> template_override = std::nullopt);

}  // namespace iom
