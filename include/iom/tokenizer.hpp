#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace iom {

struct EncodeOptions {
    bool add_special_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
};

class Tokenizer final {
public:
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    Tokenizer& operator=(Tokenizer&&) = delete;

    std::vector<std::uint32_t> encode(
            std::string_view text, EncodeOptions options) const;
    std::vector<std::uint32_t> encode_pair(
            std::string_view first, std::string_view second,
            EncodeOptions options) const;
    std::string decode(std::span<const std::uint32_t> ids,
                       DecodeOptions options) const;
    std::uint32_t bos_id() const noexcept;
    std::uint32_t eos_id() const noexcept;
    std::uint32_t unk_id() const noexcept;
    std::uint32_t tokenizer_pad_id() const noexcept;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    friend std::unique_ptr<Tokenizer> load_tokenizer(
            const std::filesystem::path& model_directory);
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Tokenizer> load_tokenizer(
        const std::filesystem::path& model_directory);

}  // namespace iom
