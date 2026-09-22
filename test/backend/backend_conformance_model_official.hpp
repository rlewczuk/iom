#pragma once

// Backend-neutral, opt-in official TinyLlama inference evidence.  Backend
// drivers create the Device and allocator; this header owns only caller input
// validation, model/session execution, comparison, and evidence publication.
// No backend runtime, registration, option, or fallback is present here.

#include <algorithm>
#include <cerrno>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>
#include <initializer_list>
#include <sys/stat.h>

#include <nlohmann/json.hpp>

#include "iom/device.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/model.hpp"
#include "iom/session.hpp"
#include "iom/token_selection.hpp"
#include "../../src/session_internal.hpp"

namespace iom_conformance {
namespace official_model_detail {

inline constexpr std::size_t kLayers = 22;
inline constexpr std::size_t kHidden = 2048;
inline constexpr std::size_t kIntermediate = 5632;
inline constexpr std::size_t kQueryHeads = 32;
inline constexpr std::size_t kKeyValueHeads = 4;
inline constexpr std::size_t kHeadDim = 64;
inline constexpr std::size_t kVocabulary = 32000;
inline constexpr std::size_t kContext = 2048;
inline constexpr std::array<std::size_t, 4> kForcedIds{3, 4, 5, 6};
inline constexpr float kAbsoluteTolerance = 0.25F;
inline constexpr float kRelativeTolerance = 0.02F;

inline std::string failure_message(std::exception_ptr error) {
    if (!error) return {};
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& exception) {
        return exception.what();
    } catch (...) {
        return "non-standard exception";
    }
}

inline std::string required_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(
                std::string(name) + " must be supplied explicitly");
    }
    return std::string(value);
}

inline std::optional<std::string> optional_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return std::nullopt;
    if (*value == '\0') {
        throw std::invalid_argument(
                std::string(name) + " must not be empty when supplied");
    }
    return std::string(value);
}

inline std::size_t parse_arena_bytes(std::string_view text) {
    if (text.empty()
            || text.find_first_not_of("0123456789") != std::string_view::npos) {
        throw std::invalid_argument(
                "IOM_TEST_MODEL_ARENA_BYTES must be a positive decimal byte "
                "count divisible by 32");
    }
    std::size_t result = 0;
    for (const char character : text) {
        const std::size_t digit = static_cast<std::size_t>(character - '0');
        if (result > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            throw std::overflow_error(
                    "IOM_TEST_MODEL_ARENA_BYTES overflows size_t");
        }
        result = result * 10 + digit;
    }
    if (result == 0 || result % 32 != 0) {
        throw std::invalid_argument(
                "IOM_TEST_MODEL_ARENA_BYTES must be a positive decimal byte "
                "count divisible by 32");
    }
    return result;
}

// Small self-contained SHA-256 implementation.  The official runner hashes
// the caller's files itself so the C++ result cannot be bound merely to an
// asserted model ID or to a wrapper's unobserved environment.
class Sha256 final {
public:
    Sha256() noexcept { reset(); }

    void update(std::span<const std::byte> bytes) noexcept {
        for (const std::byte byte : bytes) {
            buffer_[buffer_size_++] = std::to_integer<std::uint8_t>(byte);
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finish() const noexcept {
        Sha256 copy = *this;
        const std::uint64_t original_bits = copy.bit_count_
                + static_cast<std::uint64_t>(copy.buffer_size_) * 8;
        copy.buffer_[copy.buffer_size_++] = 0x80;
        if (copy.buffer_size_ > 56) {
            while (copy.buffer_size_ < 64) copy.buffer_[copy.buffer_size_++] = 0;
            copy.transform(copy.buffer_.data());
            copy.buffer_size_ = 0;
        }
        while (copy.buffer_size_ < 56) copy.buffer_[copy.buffer_size_++] = 0;
        for (int index = 7; index >= 0; --index) {
            copy.buffer_[copy.buffer_size_++] = static_cast<std::uint8_t>(
                    original_bits >> (index * 8));
        }
        copy.transform(copy.buffer_.data());
        std::array<std::uint8_t, 32> result{};
        for (std::size_t index = 0; index < copy.state_.size(); ++index) {
            for (int byte = 0; byte < 4; ++byte) {
                result[index * 4 + static_cast<std::size_t>(byte)] =
                        static_cast<std::uint8_t>(
                                copy.state_[index] >> (24 - byte * 8));
            }
        }
        return result;
    }

    [[nodiscard]] std::string hex_digest() const {
        const auto digest = finish();
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint8_t byte : digest) output << std::setw(2) << static_cast<unsigned>(byte);
        return output.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> kRoundConstants{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    static constexpr std::uint32_t rotate_right(
            std::uint32_t value, unsigned amount) noexcept {
        return (value >> amount) | (value << (32 - amount));
    }

    static constexpr std::uint32_t choose(
            std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (~x & z);
    }

    static constexpr std::uint32_t majority(
            std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    void reset() noexcept {
        state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        buffer_.fill(0);
        buffer_size_ = 0;
        bit_count_ = 0;
    }

    void transform(const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24)
                    | (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16)
                    | (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8)
                    | static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t x = words[index - 15];
            const std::uint32_t y = words[index - 2];
            const std::uint32_t s0 = rotate_right(x, 7) ^ rotate_right(x, 18)
                    ^ (x >> 3);
            const std::uint32_t s1 = rotate_right(y, 17)
                    ^ rotate_right(y, 19) ^ (y >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11)
                    ^ rotate_right(e, 25);
            const std::uint32_t temp1 = h + s1 + choose(e, f, g)
                    + kRoundConstants[index] + words[index];
            const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13)
                    ^ rotate_right(a, 22);
            const std::uint32_t temp2 = s0 + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_count_ = 0;
};

inline std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open official artifact for hashing: " + path.string());
    }
    Sha256 digest;
    std::array<std::byte, 1u << 16> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) digest.update(std::span<const std::byte>(
                buffer.data(), static_cast<std::size_t>(count)));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while hashing official artifact: " + path.string());
    }
    return digest.hex_digest();
}

inline std::size_t official_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status)) {
        throw std::invalid_argument(
                "official artifact must be a regular file: " + path.string());
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error) throw std::runtime_error("cannot stat official artifact: " + path.string());
    return size;
}

inline std::filesystem::path checked_relative_artifact(
        const std::filesystem::path& root, std::string_view relative) {
    const std::filesystem::path path(relative);
    if (path.empty() || path.is_absolute()) {
        throw std::invalid_argument("official artifact path must be relative");
    }
    for (const auto& component : path) {
        if (component.empty() || component == "." || component == "..") {
            throw std::invalid_argument("official artifact path is not normalized: " + path.string());
        }
    }
    const std::filesystem::path candidate = root / path;
    std::error_code error;
    const std::filesystem::path root_resolved = std::filesystem::canonical(root, error);
    if (error) {
        throw std::runtime_error("cannot resolve official model directory: " + root.string());
    }
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(candidate, error);
    if (error) {
        throw std::runtime_error("cannot resolve official artifact: " + candidate.string());
    }
    auto root_it = root_resolved.begin();
    auto path_it = resolved.begin();
    for (; root_it != root_resolved.end(); ++root_it, ++path_it) {
        if (path_it == resolved.end() || *root_it != *path_it) {
            throw std::invalid_argument("official artifact escapes model directory: " + path.string());
        }
    }
    if (path_it == resolved.end() && resolved != root_resolved) {
        throw std::invalid_argument("official artifact path resolves to the model directory: " + path.string());
    }
    return candidate;
}

inline std::filesystem::path normalized_absolute(
        const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path absolute =
            std::filesystem::absolute(path, error);
    if (error) {
        throw std::runtime_error(
                "cannot resolve absolute path: " + path.string());
    }
    return absolute.lexically_normal();
}

inline bool path_within(
        const std::filesystem::path& root,
        const std::filesystem::path& candidate) {
    auto root_iterator = root.begin();
    auto candidate_iterator = candidate.begin();
    while (root_iterator != root.end()
           && candidate_iterator != candidate.end()
           && *root_iterator == *candidate_iterator) {
        ++root_iterator;
        ++candidate_iterator;
    }
    return root_iterator == root.end();
}

inline void require_regular_nonsymlink(
        const std::filesystem::path& path, std::string_view label) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status)) {
        throw std::invalid_argument(
                std::string(label) + " must name a regular non-symlink file: "
                + path.string());
    }
}

struct FileIdentity {
    std::uintmax_t device;
    std::uintmax_t inode;

    bool operator==(const FileIdentity&) const = default;
};

inline std::optional<FileIdentity> available_identity(
        const std::filesystem::path& path) {
    struct stat information {};
    if (::stat(path.c_str(), &information) != 0) return std::nullopt;
    return FileIdentity{
            static_cast<std::uintmax_t>(information.st_dev),
            static_cast<std::uintmax_t>(information.st_ino)};
}

inline std::filesystem::path resolved_if_available(
        const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path resolved =
            std::filesystem::weakly_canonical(path, error);
    return error ? normalized_absolute(path) : resolved;
}

class EvidencePublication final {
public:
    EvidencePublication(
            std::filesystem::path evidence_path,
            std::filesystem::path model_directory,
            const std::filesystem::path& reference_path)
            : evidence_path_(std::move(evidence_path)),
              temporary_path_(evidence_path_.string() + ".tmp"),
              evidence_lexical_(normalized_absolute(evidence_path_)),
              evidence_resolved_(resolved_if_available(evidence_path_)),
              temporary_lexical_(normalized_absolute(temporary_path_)),
              temporary_resolved_(resolved_if_available(temporary_path_)),
              model_directory_(std::move(model_directory)),
              model_root_lexical_(normalized_absolute(model_directory_)),
              model_root_resolved_(
                      resolved_if_available(model_directory_)) {
        if (evidence_path_.empty()) {
            throw std::invalid_argument(
                    "IOM_TEST_MODEL_EVIDENCE must not be empty");
        }
        const std::filesystem::path parent = evidence_path_.parent_path();
        if (!parent.empty() && !std::filesystem::is_directory(parent)) {
            throw std::invalid_argument(
                    "official evidence parent does not exist: "
                    + parent.string());
        }
        add_protected(reference_path, "reference");
        constexpr std::array<std::string_view, 8> names{
                "config.json", "generation_config.json",
                "special_tokens_map.json", "tokenizer.json",
                "tokenizer.model", "tokenizer_config.json",
                "model.safetensors", "model.safetensors.index.json"};
        for (const std::string_view name : names) {
            add_protected(
                    model_directory_ / name,
                    "model artifact " + std::string(name));
        }
        retain_current_model_entries();
        validate();
    }

    void add_protected(
            const std::filesystem::path& path, std::string label) {
        const std::filesystem::path lexical = normalized_absolute(path);
        if (std::ranges::any_of(
                    protected_,
                    [&](const ProtectedPath& item) {
                        return item.lexical == lexical;
                    })) {
            return;
        }
        protected_.push_back({
                path,
                lexical,
                resolved_if_available(path),
                available_identity(path),
                std::move(label)});
    }

    void clear_stale() const {
        validate();
        std::error_code error;
        std::filesystem::remove(evidence_path_, error);
        if (error) {
            throw std::runtime_error(
                    "cannot clear prior official evidence: "
                    + evidence_path_.string());
        }
    }

    void validate() const {
        validate_candidate(
                evidence_path_, evidence_lexical_, evidence_resolved_);
        validate_candidate(
                temporary_path_, temporary_lexical_, temporary_resolved_);
    }

private:
    struct ProtectedPath {
        std::filesystem::path path;
        std::filesystem::path lexical;
        std::filesystem::path resolved;
        std::optional<FileIdentity> identity;
        std::string label;
    };

    void retain_current_model_entries() {
        std::error_code error;
        std::filesystem::directory_iterator iterator(
                model_directory_, error);
        const std::filesystem::directory_iterator end;
        for (; !error && iterator != end; iterator.increment(error)) {
            add_protected(
                    iterator->path(),
                    "model artifact " + iterator->path().filename().string());
        }
    }

    void validate_candidate(
            const std::filesystem::path& candidate,
            const std::filesystem::path& retained_lexical,
            const std::filesystem::path& retained_resolved) const {
        const std::filesystem::path lexical =
                normalized_absolute(candidate);
        const std::filesystem::path resolved =
                resolved_if_available(candidate);
        if (path_within(model_root_lexical_, lexical)
                || path_within(model_root_lexical_, retained_lexical)
                || path_within(model_root_resolved_, resolved)
                || path_within(model_root_resolved_, retained_resolved)) {
            throw std::invalid_argument(
                    "IOM_TEST_MODEL_EVIDENCE must be outside the caller "
                    "model directory");
        }

        std::optional<FileIdentity> destination_identity;
        struct stat information {};
        if (::lstat(candidate.c_str(), &information) == 0) {
            if (!S_ISREG(information.st_mode)) {
                throw std::invalid_argument(
                        "IOM_TEST_MODEL_EVIDENCE must name a regular "
                        "non-symlink file when present");
            }
            destination_identity = FileIdentity{
                    static_cast<std::uintmax_t>(information.st_dev),
                    static_cast<std::uintmax_t>(information.st_ino)};
        } else if (errno != ENOENT && errno != ENOTDIR) {
            throw std::runtime_error(
                    "cannot inspect official evidence destination: "
                    + candidate.string());
        }

        const auto aliases = [&](const ProtectedPath& protected_path) {
            const std::optional<FileIdentity> current =
                    available_identity(protected_path.path);
            const std::filesystem::path current_resolved =
                    resolved_if_available(protected_path.path);
            return lexical == protected_path.lexical
                    || retained_lexical == protected_path.lexical
                    || resolved == protected_path.resolved
                    || retained_resolved == protected_path.resolved
                    || resolved == current_resolved
                    || retained_resolved == current_resolved
                    || (destination_identity
                        && protected_path.identity
                        && *destination_identity
                                == *protected_path.identity)
                    || (destination_identity && current
                        && *destination_identity == *current);
        };
        for (const ProtectedPath& protected_path : protected_) {
            if (aliases(protected_path)) {
                throw std::invalid_argument(
                        "IOM_TEST_MODEL_EVIDENCE aliases protected "
                        + protected_path.label);
            }
        }

        std::error_code error;
        std::filesystem::directory_iterator iterator(
                model_directory_, error);
        const std::filesystem::directory_iterator end;
        for (; destination_identity && !error && iterator != end;
             iterator.increment(error)) {
            const std::optional<FileIdentity> current =
                    available_identity(iterator->path());
            if (current && *destination_identity == *current) {
                throw std::invalid_argument(
                        "IOM_TEST_MODEL_EVIDENCE aliases protected model "
                        "artifact");
            }
        }
    }

    std::filesystem::path evidence_path_;
    std::filesystem::path temporary_path_;
    std::filesystem::path evidence_lexical_;
    std::filesystem::path evidence_resolved_;
    std::filesystem::path temporary_lexical_;
    std::filesystem::path temporary_resolved_;
    std::filesystem::path model_directory_;
    std::filesystem::path model_root_lexical_;
    std::filesystem::path model_root_resolved_;
    std::vector<ProtectedPath> protected_;
};

inline nlohmann::json read_json(const std::filesystem::path& path,
                                std::string_view label) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot open ") + std::string(label) + ": " + path.string());
    try {
        nlohmann::json value;
        input >> value;
        return value;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(
                std::string(label) + " is malformed: " + error.what());
    }
}

inline const nlohmann::json& object_member(
        const nlohmann::json& object, std::string_view key,
        std::string_view context) {
    if (!object.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be an object");
    }
    const auto iterator = object.find(std::string(key));
    if (iterator == object.end()) {
        throw std::invalid_argument(
                std::string(context) + " is missing field " + std::string(key));
    }
    return *iterator;
}
inline void exact_fields(
        const nlohmann::json& object,
        std::initializer_list<std::string_view> expected,
        std::string_view context) {
    if (!object.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be an object");
    }
    std::set<std::string> actual_fields;
    for (const auto& item : object.items()) actual_fields.insert(item.key());
    std::set<std::string> expected_fields;
    for (const std::string_view field : expected) expected_fields.emplace(field);
    if (actual_fields != expected_fields) {
        throw std::invalid_argument(std::string(context) + " has unexpected fields");
    }
}

inline std::string string_member(
        const nlohmann::json& object, std::string_view key,
        std::string_view context, bool allow_empty = false) {
    const auto& value = object_member(object, key, context);
    if (!value.is_string() || (!allow_empty && value.get<std::string>().empty())) {
        throw std::invalid_argument(
                std::string(context) + "." + std::string(key) + " must be a non-empty string");
    }
    return value.get<std::string>();
}

inline std::size_t size_member(
        const nlohmann::json& object, std::string_view key,
        std::string_view context, bool allow_zero = true) {
    const auto& value = object_member(object, key, context);
    if ((!value.is_number_unsigned() && !value.is_number_integer())
            || value.get<std::int64_t>() < 0
            || (!allow_zero && value.get<std::uint64_t>() == 0)) {
        throw std::invalid_argument(
                std::string(context) + "." + std::string(key) + " must be a valid size");
    }
    try {
        return value.get<std::size_t>();
    } catch (const nlohmann::json::exception&) {
        throw std::overflow_error(
                std::string(context) + "." + std::string(key) + " overflows size_t");
    }
}

inline bool bool_member(
        const nlohmann::json& object, std::string_view key,
        std::string_view context) {
    const auto& value = object_member(object, key, context);
    if (!value.is_boolean()) {
        throw std::invalid_argument(
                std::string(context) + "." + std::string(key) + " must be boolean");
    }
    return value.get<bool>();
}

inline std::vector<std::size_t> token_ids(
        const nlohmann::json& value, std::string_view context) {
    if (!value.is_array()) throw std::invalid_argument(std::string(context) + " must be an ID array");
    std::vector<std::size_t> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto& item = value.at(index);
        if ((!item.is_number_unsigned() && !item.is_number_integer())
                || item.get<std::int64_t>() < 0) {
            throw std::invalid_argument(std::string(context) + " contains an invalid token ID");
        }
        const std::size_t id = item.get<std::size_t>();
        if (id >= kVocabulary) {
            throw std::invalid_argument(std::string(context) + " contains an out-of-range token ID");
        }
        result.push_back(id);
    }
    return result;
}

inline std::vector<float> finite_logits(
        const nlohmann::json& value, std::string_view context,
        std::size_t vocabulary = kVocabulary) {
    if (vocabulary == 0 || !value.is_array() || value.size() != vocabulary) {
        throw std::invalid_argument(
                std::string(context) + " must contain the full vocabulary");
    }
    std::vector<float> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto& item = value.at(index);
        if (!item.is_number_float() && !item.is_number_integer()
                && !item.is_number_unsigned()) {
            throw std::invalid_argument(
                    std::string(context) + " contains a non-number");
        }
        const double number = item.get<double>();
        if (!std::isfinite(number)) {
            throw std::invalid_argument(
                    std::string(context) + " contains a non-finite value");
        }
        result.push_back(static_cast<float>(number));
        if (!std::isfinite(result.back())) {
            throw std::invalid_argument(
                    std::string(context) + " contains an unrepresentable value");
        }
    }
    return result;
}

inline std::size_t lowest_argmax(std::span<const float> values) {
    if (values.empty()) throw std::invalid_argument("official logits are empty");
    std::size_t winner = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[winner]) winner = index;
    }
    return winner;
}

inline bool reference_margin_stable(
        std::span<const float> values,
        float absolute_tolerance = kAbsoluteTolerance,
        float relative_tolerance = kRelativeTolerance) {
    const std::size_t winner = lowest_argmax(values);
    const double winner_bound = absolute_tolerance
            + relative_tolerance
                    * std::abs(static_cast<double>(values[winner]));
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index == winner) continue;
        const double bound = absolute_tolerance
                + relative_tolerance
                        * std::abs(static_cast<double>(values[index]));
        if (!(static_cast<double>(values[winner]) - winner_bound
              > static_cast<double>(values[index]) + bound)) {
            return false;
        }
    }
    return true;
}

inline void validate_lower_sha256(
        const nlohmann::json& value, std::string_view key,
        std::string_view context) {
    const std::string digest = string_member(value, key, context);
    if (digest.size() != 64
            || std::any_of(digest.begin(), digest.end(), [](char character) {
                   return !(character >= '0' && character <= '9')
                           && !(character >= 'a' && character <= 'f');
               })) {
        throw std::invalid_argument(std::string(context) + "." + std::string(key)
                + " must be lowercase SHA-256");
    }
}
inline void compare_logits(
        std::span<const float> actual, std::span<const float> reference,
        std::string_view context,
        float absolute_tolerance = kAbsoluteTolerance,
        float relative_tolerance = kRelativeTolerance) {
    if (actual.size() != reference.size()) {
        throw std::runtime_error(std::string(context) + " vocabulary size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
            throw std::runtime_error(std::string(context) + " contains a non-finite value");
        }
        const double bound = absolute_tolerance
                + relative_tolerance
                        * std::abs(static_cast<double>(reference[index]));
        if (std::abs(static_cast<double>(actual[index]) - reference[index]) > bound) {
            throw std::runtime_error(
                    std::string(context) + " mismatch at vocabulary index "
                    + std::to_string(index));
        }
    }
}

inline std::string backend_name(iom::BackendKind backend) {
    switch (backend) {
    case iom::BackendKind::CPU:
        return "cpu";
    case iom::BackendKind::CUDA:
        return "cuda";
    case iom::BackendKind::ROCM:
        return "rocm";
    case iom::BackendKind::SYCL:
        return "sycl";
    }
    return "unknown";
}

inline std::string stop_name(iom::GenerationStopReason reason) {
    switch (reason) {
    case iom::GenerationStopReason::eos:
        return "eos";
    case iom::GenerationStopReason::max_new_tokens:
        return "max_new_tokens";
    case iom::GenerationStopReason::context_capacity:
        return "context_exhaustion";
    }
    return "unknown";
}

inline nlohmann::json invocation_record() {
    std::ifstream command_line("/proc/self/cmdline", std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(command_line), {}};
    nlohmann::json arguments = nlohmann::json::array();
    std::size_t begin = 0;
    while (begin < bytes.size()) {
        const std::size_t end = bytes.find('\0', begin);
        arguments.push_back(bytes.substr(
                begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (arguments.empty()) arguments.push_back("<unavailable>");
    return arguments;
}

// A synchronous selector that records every complete BF16 vocabulary row and
// its borrowed history.  It is test-only: production token selection remains
// the existing GreedyTokenSelector, while forced continuation is explicitly
// labeled and never presented as greedy-token evidence.
class RecordingSelector final : public iom::TokenSelector {
public:
    explicit RecordingSelector(std::span<const std::size_t> forced = {})
        : forced_(forced.begin(), forced.end()) {
        selected.reserve(forced_.empty() ? 4 : forced_.size());
        histories.reserve(forced_.empty() ? 4 : forced_.size());
        logits.reserve(forced_.empty() ? 4 : forced_.size());
    }

    [[nodiscard]] iom::TokenSelectorScratchRequirements scratch_requirements(
            const iom::TensorView& view, std::size_t vocabulary) const override {
        if (vocabulary == 0 || view.spec().data_type != iom::DataType::BF16
                || view.spec().shape.rank() != 2
                || view.spec().shape.dimensions()[0] != 1
                || view.spec().shape.dimensions()[1] != vocabulary) {
            throw std::invalid_argument("official recording selector received invalid logits");
        }
        return iom::TokenSelectorScratchRequirements{
                vocabulary * sizeof(std::uint16_t),
                view.copy_to_host_workspace_requirements()};
    }

    [[nodiscard]] std::size_t select(
            iom::DeviceOps& queue, const iom::TensorView& view,
            std::size_t vocabulary, iom::oid producer,
            std::span<const std::size_t> history,
            iom::TokenSelectorScratch scratch) override {
        if (producer <= 0) throw std::invalid_argument("official selector producer is not positive");
        queue.wait(producer);
        const std::size_t bytes = vocabulary * sizeof(std::uint16_t);
        if (scratch.host.size() < bytes) throw std::invalid_argument("official selector host scratch is too small");
        view.copy_to_host(scratch.host.first(bytes), scratch.device);
        std::vector<float> row(vocabulary);
        for (std::size_t index = 0; index < vocabulary; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, scratch.host.data() + index * sizeof(bits), sizeof(bits));
            row[index] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
            if (!std::isfinite(row[index])) throw std::runtime_error("official selector observed non-finite logits");
        }
        logits.push_back(row);
        histories.emplace_back(history.begin(), history.end());
        const std::size_t index = selected.size();
        const std::size_t id = forced_.empty()
                ? lowest_argmax(std::span<const float>(row))
                : (index < forced_.size() ? forced_[index]
                                           : throw std::logic_error("official forced continuation exhausted"));
        if (id >= vocabulary) throw std::invalid_argument("official selector returned an out-of-range ID");
        selected.push_back(id);
        return id;
    }

    std::vector<std::vector<float>> logits;
    std::vector<std::vector<std::size_t>> histories;
    std::vector<std::size_t> selected;

private:
    std::vector<std::size_t> forced_;
};

struct OfficialCase final {
    std::string name;
    nlohmann::json record;
    std::vector<std::size_t> prompt;
    std::vector<std::size_t> decode;
    std::size_t limit = 0;
    std::string mode;
    std::string policy;
    std::size_t layers = kLayers;
    std::size_t vocabulary = kVocabulary;
    float absolute_tolerance = kAbsoluteTolerance;
    float relative_tolerance = kRelativeTolerance;
};

inline std::vector<OfficialCase> validate_reference_pack(
        const nlohmann::json& pack, const std::string& artifact_id) {
    if (!pack.is_object() || pack.size() != 7
            || !pack.contains("schema_version") || !pack.contains("kind")
            || !pack.contains("provenance") || !pack.contains("tolerances")
            || !pack.contains("model_config") || !pack.contains("artifact_id")
            || !pack.contains("cases")) {
        throw std::invalid_argument("official reference pack has an invalid top-level schema");
    }
    if (pack.at("schema_version") != 1 || pack.at("kind") != "official") {
        throw std::invalid_argument("official reference pack schema_version/kind mismatch");
    }
    if (!pack.at("artifact_id").is_string()
            || pack.at("artifact_id").get<std::string>() != artifact_id) {
        throw std::invalid_argument("official artifact_id differs from IOM_TEST_MODEL_ID");
    }
    const nlohmann::json expected_tolerances = {
            {"absolute", 0.25},
            {"relative", 0.02},
            {"formula", "abs(actual-ref) <= 0.25 + 0.02*abs(ref)"},
            {"tie_policy", "lowest-id"},
            {"exact_token", "reference-margin-certified-only"}};
    if (pack.at("tolerances") != expected_tolerances) {
        throw std::invalid_argument("official reference tolerance policy is not frozen");
    }
    const auto& provenance = pack.at("provenance");
    const auto& runtime = object_member(provenance, "runtime", "official provenance");
    exact_fields(runtime, {"implementation", "version"}, "official provenance.runtime");
    if (string_member(runtime, "implementation", "official provenance.runtime") != "CPython"
            || string_member(runtime, "version", "official provenance.runtime") != "3.11.16") {
        throw std::invalid_argument("official provenance runtime pin mismatch");
    }
    const auto& packages = object_member(provenance, "packages", "official provenance");
    const std::array<std::pair<std::string_view, std::string_view>, 7> package_pins{{
            {"jinja2", "3.1.2"}, {"numpy", "1.26.4"}, {"safetensors", "0.4.1"},
            {"sentencepiece", "0.1.99"}, {"tokenizers", "0.14.1"},
            {"torch", "2.1.2"}, {"transformers", "4.35.0"}}};
    if (!packages.is_object() || packages.size() != package_pins.size()) {
        throw std::invalid_argument("official provenance package set mismatch");
    }
    for (const auto [package, version] : package_pins) {
        const std::string actual = string_member(packages, package, "official provenance.packages");
        if (actual != version
                && (actual.size() <= version.size()
                    || actual.compare(0, version.size(), version) != 0
                    || actual[version.size()] != '+')) {
            throw std::invalid_argument("official provenance package pin mismatch");
        }
    }
    const auto& precision = object_member(provenance, "precision", "official provenance");
    exact_fields(precision, {"weights", "activations", "attention", "logits_storage",
                             "reference_arithmetic", "device"},
                 "official provenance.precision");
    if (precision != nlohmann::json{
                {"weights", "BF16"}, {"activations", "BF16"}, {"attention", "eager"},
                {"logits_storage", "BF16"}, {"reference_arithmetic", "FP32"},
                {"device", "cpu"}}) {
        throw std::invalid_argument("official provenance precision policy mismatch");
    }
    validate_lower_sha256(provenance, "generator_sha256", "official provenance");
    validate_lower_sha256(provenance, "exporter_sha256", "official provenance");
    const auto& command = object_member(provenance, "command", "official provenance");
    if (!command.is_array() || command.empty()
            || std::any_of(command.begin(), command.end(), [](const nlohmann::json& value) {
                   return !value.is_string() || value.get<std::string>().empty();
               })) {
        throw std::invalid_argument("official provenance command must be a non-empty argv");
    }
    const std::string digest = string_member(
            provenance, "case_payload_sha256", "official provenance");
    if (digest.size() != 64
            || std::any_of(digest.begin(), digest.end(), [](char value) {
                   return !(value >= '0' && value <= '9')
                           && !(value >= 'a' && value <= 'f');
               })) {
        throw std::invalid_argument("official case payload digest is not lowercase SHA-256");
    }
    const nlohmann::json& model_config = object_member(
            pack, "model_config", "official reference pack");
    const std::string model_id = string_member(
            model_config, "model_id", "official model_config");
    if (string_member(provenance, "model_id", "official provenance") != model_id
            || string_member(provenance, "revision", "official provenance") != artifact_id) {
        throw std::invalid_argument("official provenance model/revision identity mismatch");
    }
    const auto require_geometry = [&](std::string_view field, std::size_t value) {
        if (size_member(model_config, field, "official model_config") != value) {
            throw std::invalid_argument("official model_config geometry mismatch for " + std::string(field));
        }
    };
    require_geometry("num_hidden_layers", kLayers);
    require_geometry("hidden_size", kHidden);
    require_geometry("intermediate_size", kIntermediate);
    require_geometry("num_attention_heads", kQueryHeads);
    require_geometry("num_key_value_heads", kKeyValueHeads);
    require_geometry("head_dim", kHeadDim);
    require_geometry("vocab_size", kVocabulary);
    require_geometry("max_position_embeddings", kContext);
    if (string_member(model_config, "model_type", "official model_config") != "llama"
            || bool_member(model_config, "tie_word_embeddings", "official model_config")
            || string_member(model_config, "torch_dtype", "official model_config") != "bfloat16") {
        throw std::invalid_argument("official model_config requires untied BF16 Llama geometry");
    }
    const auto& cases = pack.at("cases");
    if (!cases.is_array() || cases.size() != 5) {
        throw std::invalid_argument("official reference pack must contain five cases");
    }
    static constexpr std::array<std::string_view, 5> names{
            "raw-production-greedy", "raw-fixed-reference-continuation",
            "chat-production-greedy", "chat-fixed-reference-continuation",
            "zero-new-token"};
    std::vector<OfficialCase> result;
    result.reserve(cases.size());
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& record = cases.at(case_index);
        const std::string context = "official case " + std::to_string(case_index);
        exact_fields(
                record, {"id", "input", "snapshots", "expected_result"},
                context);
        const std::string name = string_member(record, "id", context);
        if (name != names[case_index]) {
            throw std::invalid_argument(context + " id/order mismatch");
        }
        const auto& input = object_member(record, "input", context);
        const std::string mode = string_member(input, "mode", context + ".input");
        const std::string policy = string_member(input, "selection_policy", context + ".input");
        if (mode == "raw") {
            exact_fields(input, {"mode", "selection_policy", "prompt_ids",
                                 "rendered_utf8", "positions", "decode_ids",
                                 "max_new_tokens", "bos_policy", "text"},
                         context + ".input");
        } else {
            exact_fields(input, {"mode", "selection_policy", "prompt_ids",
                                 "rendered_utf8", "positions", "decode_ids",
                                 "max_new_tokens", "bos_policy", "messages"},
                         context + ".input");
        }
        if ((name.starts_with("chat-") && mode != "chat")
                || (name.starts_with("raw-") && mode != "raw")) {
            throw std::invalid_argument(context + " mode does not match its fixed case");
        }
        OfficialCase parsed{name, record, token_ids(
                object_member(input, "prompt_ids", context + ".input"),
                context + ".input.prompt_ids"),
                token_ids(object_member(input, "decode_ids", context + ".input"),
                          context + ".input.decode_ids"),
                size_member(input, "max_new_tokens", context + ".input"), mode,
                policy};
        const auto& bos_policy = object_member(input, "bos_policy", context + ".input");
        exact_fields(bos_policy, {"add_special_tokens", "require_bos", "bos_token_id"},
                     context + ".input.bos_policy");
        if (bos_policy.size() != 3
                || !bos_policy.contains("add_special_tokens")
                || !bos_policy.contains("require_bos")
                || !bos_policy.contains("bos_token_id")
                || size_member(bos_policy, "bos_token_id", context + ".input.bos_policy") != 1) {
            throw std::invalid_argument(context + " BOS policy is malformed");
        }
        const bool expected_special = mode == "raw";
        if (bool_member(bos_policy, "add_special_tokens", context + ".input.bos_policy") != expected_special
                || bool_member(bos_policy, "require_bos", context + ".input.bos_policy") != expected_special) {
            throw std::invalid_argument(context + " BOS policy differs from the fixed prompt policy");
        }
        const auto& positions = object_member(input, "positions", context + ".input");
        if (!positions.is_array() || positions.size() != parsed.prompt.size()) {
            throw std::invalid_argument(context + " prompt positions have the wrong shape");
        }
        for (std::size_t index = 0; index < positions.size(); ++index) {
            const auto& value = positions.at(index);
            if (!value.is_number_unsigned() || value.get<std::size_t>() != index) {
                throw std::invalid_argument(context + " prompt positions are not contiguous");
            }
        }
        const auto& rendered = object_member(input, "rendered_utf8", context + ".input");
        if (!rendered.is_array()) throw std::invalid_argument(context + " rendered_utf8 is not an array");
        for (const auto& byte : rendered) {
            if (!byte.is_number_unsigned() || byte.get<std::size_t>() > 255) {
                throw std::invalid_argument(context + " rendered_utf8 contains an invalid byte");
            }
        }
        if (mode == "raw") {
            if (string_member(input, "text", context + ".input") != "The capital of France is") {
                throw std::invalid_argument(context + " raw text differs from frozen policy");
            }
            const std::string expected_raw = "The capital of France is";
            if (rendered.size() != expected_raw.size()) {
                throw std::invalid_argument(context + " raw rendered bytes differ from frozen policy");
            }
            for (std::size_t index = 0; index < rendered.size(); ++index) {
                if (rendered.at(index).get<unsigned>() != static_cast<unsigned char>(expected_raw[index])) {
                    throw std::invalid_argument(context + " raw rendered bytes differ from frozen policy");
                }
            }
            if (parsed.prompt.empty() || parsed.prompt.front() != 1) {
                throw std::invalid_argument(context + " raw prompt does not record the required BOS");
            }
        } else {
            const auto& messages = object_member(input, "messages", context + ".input");
            if (messages != nlohmann::json::array({
                        nlohmann::json{{"role", "user"}, {"content", "Hello."}}})) {
                throw std::invalid_argument(context + " chat messages differ from frozen policy");
            }
            const std::string expected_rendered = "<|user|>\nHello.</s>\n<|assistant|>\n";
            std::string actual_rendered;
            actual_rendered.reserve(rendered.size());
            for (const auto& byte : rendered) actual_rendered.push_back(static_cast<char>(byte.get<unsigned>()));
            if (actual_rendered != expected_rendered) throw std::invalid_argument(context + " chat rendered bytes differ from frozen policy");
            if (parsed.prompt.empty() || parsed.prompt.front() == 1) {
                throw std::invalid_argument(context + " chat prompt unexpectedly has a BOS");
            }
        }
        if (name == "zero-new-token") {
            if (policy != "none" || !parsed.decode.empty() || parsed.limit != 0) throw std::invalid_argument(context + " zero policy mismatch");
        } else if (name.ends_with("fixed-reference-continuation")) {
            if (policy != "fixed-reference-continuation" || parsed.limit != 4
                    || parsed.decode != std::vector<std::size_t>(kForcedIds.begin(), kForcedIds.end())) {
                throw std::invalid_argument(context + " forced continuation policy mismatch");
            }
        } else if (policy != "production-greedy" || parsed.limit != 4
                   || parsed.decode.empty() || parsed.decode.size() > 4) {
            throw std::invalid_argument(context + " production policy mismatch");
        }
        if (parsed.policy == "production-greedy") {
            const auto eos = std::find(
                    parsed.decode.begin(), parsed.decode.end(), 2);
            if (eos != parsed.decode.end()
                    && std::next(eos) != parsed.decode.end()) {
                throw std::invalid_argument(
                        context + " contains output after EOS");
            }
            if (parsed.decode.size() < parsed.limit
                    && eos == parsed.decode.end()
                    && parsed.prompt.size() + parsed.decode.size()
                            < kContext) {
                throw std::invalid_argument(
                        context + " has no valid stop condition");
            }
        }
        const auto& snapshots = object_member(record, "snapshots", context);
        if (!snapshots.is_array() || snapshots.size() != parsed.decode.size()) {
            throw std::invalid_argument(context + " snapshot count does not match decode IDs");
        }
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            const auto& snapshot = snapshots.at(index);
            const std::string snapshot_context = context + ".snapshots[" + std::to_string(index) + "]";
            exact_fields(snapshot, {"phase", "prefix_ids", "position_start",
                                    "run_length", "absolute_positions", "logits",
                                    "greedy_id", "selected_id", "margin_stable"},
                         snapshot_context);
            const std::string phase = string_member(snapshot, "phase", snapshot_context);
            const std::string expected_phase = index == 0 ? "prefill" : "cached-decode";
            if (phase != expected_phase) {
                throw std::invalid_argument(snapshot_context + " phase mismatch");
            }
            std::vector<std::size_t> prefix = parsed.prompt;
            prefix.insert(prefix.end(), parsed.decode.begin(),
                          parsed.decode.begin() + static_cast<std::ptrdiff_t>(index));
            if (token_ids(object_member(snapshot, "prefix_ids", snapshot_context),
                          snapshot_context + ".prefix_ids") != prefix) {
                throw std::invalid_argument(snapshot_context + " prefix mismatch");
            }
            const std::size_t expected_start = index == 0 ? 0 : prefix.size() - 1;
            const std::size_t expected_length = index == 0 ? prefix.size() : 1;
            if (size_member(snapshot, "position_start", snapshot_context) != expected_start
                    || size_member(snapshot, "run_length", snapshot_context, false) != expected_length) {
                throw std::invalid_argument(snapshot_context + " position window mismatch");
            }
            std::vector<std::size_t> expected_positions;
            for (std::size_t position = 0; position < expected_length; ++position) {
                expected_positions.push_back(expected_start + position);
            }
            if (token_ids(object_member(snapshot, "absolute_positions", snapshot_context),
                          snapshot_context + ".absolute_positions") != expected_positions) {
                throw std::invalid_argument(snapshot_context + " absolute positions mismatch");
            }
            const std::vector<float> reference = finite_logits(
                    object_member(snapshot, "logits", snapshot_context),
                    snapshot_context + ".logits");
            const std::size_t greedy = size_member(snapshot, "greedy_id", snapshot_context);
            const std::size_t selected = size_member(snapshot, "selected_id", snapshot_context);
            if (greedy != lowest_argmax(reference) || selected != parsed.decode[index]) {
                throw std::invalid_argument(snapshot_context + " greedy/selected ID mismatch");
            }
            if (bool_member(snapshot, "margin_stable", snapshot_context)
                    != reference_margin_stable(reference)) {
                throw std::invalid_argument(snapshot_context + " margin policy mismatch");
            }
        }
        const auto& expected = object_member(record, "expected_result", context);
        const std::string expected_context = context + ".expected_result";
        exact_fields(expected, {"token_ids", "stop_reason", "initialized_kv_length"},
                     expected_context);
        if (token_ids(object_member(expected, "token_ids", expected_context),
                      expected_context + ".token_ids") != parsed.decode) {
            throw std::invalid_argument(context + " expected token IDs mismatch");
        }
        const std::string expected_stop = name == "zero-new-token"
                ? "max_new_tokens"
                : (parsed.policy == "fixed-reference-continuation"
                           ? "max_new_tokens"
                           : (!parsed.decode.empty() && parsed.decode.back() == 2
                                      ? "eos"
                                      : (parsed.decode.size() == parsed.limit
                                                 ? "max_new_tokens"
                                                 : "context_exhaustion")));
        if (string_member(expected, "stop_reason", expected_context) != expected_stop
                || size_member(expected, "initialized_kv_length", expected_context)
                           != (parsed.decode.empty()
                                       ? 0
                                       : parsed.prompt.size()
                                               + parsed.decode.size() - 1)) {
            throw std::invalid_argument(
                    context + " expected stop/KV state mismatch");
        }
        result.push_back(std::move(parsed));
    }
    return result;
}

inline void validate_artifacts(
        const std::filesystem::path& model_directory,
        const nlohmann::json& provenance) {
    const std::array<std::string_view, 6> required{
            "config.json", "generation_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json"};
    std::set<std::string> actual_names;
    for (const std::string_view name : required) actual_names.emplace(name);
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(model_directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        if (name.ends_with(".safetensors")) actual_names.insert(name);
    }
    if (error) throw std::runtime_error("cannot enumerate official model directory: " + model_directory.string());
    const std::filesystem::path index = model_directory / "model.safetensors.index.json";
    if (std::filesystem::exists(index) || std::filesystem::is_symlink(index)) actual_names.insert(index.filename().string());
    if (actual_names.size() == required.size()) throw std::invalid_argument("official model has no SafeTensors shard");

    const auto& records = object_member(provenance, "artifacts", "official provenance");
    if (!records.is_array() || records.empty()) throw std::invalid_argument("official provenance artifacts are missing");
    std::set<std::string> recorded_names;
    for (std::size_t index_record = 0; index_record < records.size(); ++index_record) {
        const auto& record = records.at(index_record);
        const std::string context = "official artifact[" + std::to_string(index_record) + "]";
        exact_fields(record, {"path", "size", "sha256"}, context);
        const std::string relative = string_member(record, "path", context);
        const std::filesystem::path checked = checked_relative_artifact(model_directory, relative);
        if (!recorded_names.insert(relative).second) throw std::invalid_argument(context + " is duplicated");
        const std::size_t expected_size = size_member(record, "size", context);
        const std::string expected_hash = string_member(record, "sha256", context);
        if (expected_hash.size() != 64 || std::any_of(expected_hash.begin(), expected_hash.end(), [](char value) {
                    return !std::isdigit(static_cast<unsigned char>(value))
                            && (value < 'a' || value > 'f');
                })) {
            throw std::invalid_argument(context + " has an invalid SHA-256");
        }
        if (official_file_size(checked) != expected_size || sha256_file(checked) != expected_hash) {
            throw std::runtime_error("official artifact identity mismatch for " + relative);
        }
    }
    if (recorded_names != actual_names) throw std::invalid_argument("official artifact inventory differs from caller model directory");
}

inline void validate_model_geometry(
        const iom::TinyLlamaConfig& config, const nlohmann::json& model_config) {
    const auto check = [](std::size_t actual, std::size_t expected, const char* name) {
        if (actual != expected) throw std::invalid_argument(std::string("official model geometry mismatch for ") + name);
    };
    check(config.num_hidden_layers, kLayers, "num_hidden_layers");
    check(config.hidden_size, kHidden, "hidden_size");
    check(config.intermediate_size, kIntermediate, "intermediate_size");
    check(config.num_attention_heads, kQueryHeads, "num_attention_heads");
    check(config.num_key_value_heads, kKeyValueHeads, "num_key_value_heads");
    check(config.head_dim, kHeadDim, "head_dim");
    check(config.vocab_size, kVocabulary, "vocab_size");
    check(config.max_position_embeddings, kContext, "max_position_embeddings");
    if (model_config.at("model_type") != "llama"
            || model_config.at("tie_word_embeddings") != false
            || model_config.at("torch_dtype") != "bfloat16") {
        throw std::invalid_argument("official model configuration requires untied BF16 Llama");
    }
}

struct ModeObservation final {
    std::vector<std::vector<float>> logits;
    std::vector<std::vector<std::size_t>> histories;
    std::vector<std::size_t> selected;
    std::vector<std::size_t> result_ids;
    std::vector<std::size_t> history_after;
    std::vector<std::size_t> cache_lengths;
    iom::GenerationStopReason stop_reason = iom::GenerationStopReason::max_new_tokens;
    std::string text;
    std::size_t request_length = 0;
    bool poisoned = false;
    std::size_t accepted_operations = 0;
    bool trace_prepared = false;
    std::uint64_t trace_rows_dropped = 0;
    std::optional<iom::InferenceSnapshot> snapshot;
    std::vector<iom::InferenceTraceRecord> operations;
};

inline void compare_snapshots(
        const iom::InferenceSnapshot& lhs,
        const iom::InferenceSnapshot& rhs,
        std::string_view context) {
    const auto same_phase = [&](iom::ObservationState left, iom::ObservationState right, const char* field) {
        if (left != right) throw std::runtime_error(std::string(context) + " metric " + field + " differs");
    };
    same_phase(lhs.load.state, rhs.load.state, "load");
    same_phase(lhs.admitted.tokenization.state, rhs.admitted.tokenization.state, "tokenization");
    same_phase(lhs.admitted.prefill.state, rhs.admitted.prefill.state, "prefill");
    same_phase(lhs.admitted.decode.state, rhs.admitted.decode.state, "decode");
    if (lhs.request_admitted != rhs.request_admitted
            || lhs.attempt.outcome != rhs.attempt.outcome
            || lhs.admitted.generated_tokens != rhs.admitted.generated_tokens
            || lhs.admitted.decode_forward_count != rhs.admitted.decode_forward_count
            || lhs.admitted.decode_token_count != rhs.admitted.decode_token_count
            || lhs.admitted.stop_reason_valid != rhs.admitted.stop_reason_valid
            || lhs.admitted.time_to_first_token_valid != rhs.admitted.time_to_first_token_valid
            || lhs.admitted.decode_throughput_valid != rhs.admitted.decode_throughput_valid) {
        throw std::runtime_error(std::string(context) + " scalar metric parity differs");
    }
    if (lhs.admitted.stop_reason_valid && lhs.admitted.stop_reason != rhs.admitted.stop_reason) {
        throw std::runtime_error(std::string(context) + " stop metric parity differs");
    }
    if (static_cast<bool>(lhs.attempt.failure) != static_cast<bool>(rhs.attempt.failure)) {
        throw std::runtime_error(std::string(context) + " failure metric parity differs");
    }
}

inline ModeObservation run_mode(
        iom::Device& device, const std::filesystem::path& model_directory,
        const OfficialCase& test_case, int mode) {
    std::unique_ptr<iom::InferenceMetrics> recorder;
    if (mode != 0) recorder = std::make_unique<iom::InferenceMetrics>();
    const bool forced = test_case.policy == "fixed-reference-continuation";
    std::vector<std::size_t> forced_ids = test_case.decode;
    auto selector = std::make_unique<RecordingSelector>(
            forced ? std::span<const std::size_t>(forced_ids)
                   : std::span<const std::size_t>{});
    RecordingSelector* const selector_state = selector.get();
    auto session = iom::load_tinyllama_session(
            model_directory, device, std::move(selector), recorder.get());
    if (!session) {
        throw std::runtime_error("official session factory returned null");
    }
    if (mode == 2) session->prepare_operation_trace();

    // Raw and chat prompts pass through the actual immutable formatter and
    // tokenizer owners.  The chat encoding below first proves the frozen
    // no-BOS prompt IDs; generate_chat then exercises that same supported
    // production policy.  The tokens mode is an internal seam used only by
    // the genuine tiny-fixture offline proof.
    const std::array<iom::ChatMessageView, 1> chat_messages{
            iom::ChatMessageView{"user", "Hello."}};
    const std::string raw_text = test_case.mode == "raw"
            ? string_member(
                      test_case.record.at("input"), "text",
                      test_case.name, true)
            : std::string{};
    const std::vector<std::uint32_t> encoded = [&] {
        if (test_case.mode == "raw") {
            return session->tokenizer().encode(
                    raw_text, iom::EncodeOptions{true});
        }
        if (test_case.mode == "tokens") {
            std::vector<std::uint32_t> result;
            result.reserve(test_case.prompt.size());
            for (const std::size_t id : test_case.prompt) {
                result.push_back(static_cast<std::uint32_t>(id));
            }
            return result;
        }
        if (test_case.mode != "chat") {
            throw std::invalid_argument(
                    test_case.name + " has an unsupported execution mode");
        }
        const std::string rendered =
                session->formatter().format(chat_messages, true);
        const auto& rendered_bytes =
                test_case.record.at("input").at("rendered_utf8");
        if (rendered.size() != rendered_bytes.size()) {
            throw std::runtime_error(
                    test_case.name + " rendered chat byte length mismatch");
        }
        for (std::size_t index = 0; index < rendered.size(); ++index) {
            if (static_cast<unsigned char>(rendered[index])
                    != rendered_bytes.at(index).get<unsigned>()) {
                throw std::runtime_error(
                        test_case.name + " rendered chat bytes mismatch");
            }
        }
        return session->tokenizer().encode(
                rendered, iom::EncodeOptions{false});
    }();
    if (encoded.size() != test_case.prompt.size()) {
        throw std::runtime_error(
                test_case.name + " prompt ID count mismatch");
    }
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        if (encoded[index] != test_case.prompt[index]) {
            throw std::runtime_error(
                    test_case.name + " prompt IDs mismatch");
        }
    }

    std::vector<std::size_t> result_ids;
    iom::GenerationStopReason result_stop =
            iom::GenerationStopReason::max_new_tokens;
    std::string result_text;
    if (test_case.mode == "raw") {
        const iom::GenerationResult result =
                session->generate_raw(raw_text, test_case.limit);
        result_ids.assign(result.token_ids.begin(), result.token_ids.end());
        result_stop = result.stop_reason;
        result_text = result.text;
    } else if (test_case.mode == "chat") {
        const iom::GenerationResult result =
                session->generate_chat(chat_messages, test_case.limit);
        result_ids.assign(result.token_ids.begin(), result.token_ids.end());
        result_stop = result.stop_reason;
        result_text = result.text;
    } else {
        const iom::TokenGenerationResult result =
                session->generate_tokens(test_case.prompt, test_case.limit);
        result_ids.assign(result.token_ids.begin(), result.token_ids.end());
        result_stop = result.stop_reason;
    }
    std::vector<std::uint32_t> decoded_ids;
    decoded_ids.reserve(result_ids.size());
    for (const std::size_t id : result_ids) {
        decoded_ids.push_back(static_cast<std::uint32_t>(id));
    }
    const std::string decoded = session->tokenizer().decode(
            decoded_ids, iom::DecodeOptions{true});
    if (test_case.mode != "tokens" && decoded != result_text) {
        throw std::runtime_error(
                test_case.name + " decoded generated text mismatch");
    }
    if (test_case.mode == "tokens") result_text = decoded;
    if (result_text.find("<s>") != std::string::npos
            || result_text.find("</s>") != std::string::npos) {
        throw std::runtime_error(
                test_case.name + " special-token text policy mismatch");
    }

    ModeObservation observed;
    observed.result_ids = std::move(result_ids);
    observed.stop_reason = result_stop;
    observed.text = std::move(result_text);
    observed.selected = selector_state->selected;
    observed.logits = selector_state->logits;
    observed.histories = selector_state->histories;
    observed.request_length = session->request_length();
    observed.history_after =
            iom::session_detail::SessionAccess::history(*session);
    for (const auto& cache :
         iom::session_detail::SessionAccess::caches(*session)) {
        observed.cache_lengths.push_back(cache.initialized_length);
    }
    observed.poisoned = session->poisoned();
    observed.trace_prepared = mode == 2;
    observed.accepted_operations =
            iom::session_detail::SessionAccess::accepted(*session).size();
    session.reset();
    if (recorder) {
        observed.snapshot = recorder->snapshot();
        observed.trace_rows_dropped =
                observed.snapshot->trace_rows_dropped;
        const auto rows = recorder->operations();
        observed.operations.assign(rows.begin(), rows.end());
    }
    return observed;
}

inline void compare_case(
        const OfficialCase& test_case,
        const std::array<ModeObservation, 3>& observations,
        nlohmann::json& evidence) {
    const ModeObservation& disabled = observations[0];
    const ModeObservation& metrics = observations[1];
    const ModeObservation& traced = observations[2];
    for (std::size_t index = 1; index < observations.size(); ++index) {
        if (disabled.selected != observations[index].selected
                || disabled.histories != observations[index].histories
                || disabled.logits != observations[index].logits
                || disabled.result_ids != observations[index].result_ids
                || disabled.history_after != observations[index].history_after
                || disabled.cache_lengths != observations[index].cache_lengths
                || disabled.stop_reason != observations[index].stop_reason
                || disabled.text != observations[index].text
                || disabled.poisoned != observations[index].poisoned
                || disabled.accepted_operations
                        != observations[index].accepted_operations) {
            throw std::runtime_error(
                    test_case.name + " instrumentation result parity mismatch");
        }
    }

    const auto& snapshots = test_case.record.at("snapshots");
    bool same_history = true;
    for (std::size_t index = 0; index < disabled.logits.size(); ++index) {
        if (index >= snapshots.size()) {
            throw std::runtime_error(
                    test_case.name + " produced more forwards than reference");
        }
        const std::vector<std::size_t> expected_history = token_ids(
                snapshots.at(index).at("prefix_ids"),
                test_case.name + ".snapshot.prefix_ids");
        if (disabled.histories[index] != expected_history) {
            throw std::runtime_error(
                    test_case.name
                    + " has an unexplained selector-history divergence");
        }
        const std::vector<float> reference = finite_logits(
                snapshots.at(index).at("logits"),
                test_case.name + ".snapshot.logits", test_case.vocabulary);
        compare_logits(
                disabled.logits[index], reference,
                test_case.name + ".logits", test_case.absolute_tolerance,
                test_case.relative_tolerance);
        const bool stable =
                snapshots.at(index).at("margin_stable").get<bool>();
        if (disabled.selected[index] != test_case.decode[index]) {
            if (test_case.policy == "fixed-reference-continuation" || stable) {
                throw std::runtime_error(
                        test_case.name
                        + " selected ID disagrees with the frozen policy");
            }
            // The reference margin does not certify this production-greedy
            // token.  This identical-prefix row was still compared in full,
            // but later rows belong to different histories and are not
            // compared or represented as failures.
            same_history = false;
            break;
        }
    }
    if (same_history && disabled.logits.size() != snapshots.size()) {
        throw std::runtime_error(
                test_case.name + " produced fewer forwards than reference");
    }
    if (test_case.policy == "fixed-reference-continuation"
            && (!same_history || disabled.selected != test_case.decode
                || disabled.result_ids != test_case.decode)) {
        throw std::runtime_error(
                test_case.name + " forced continuation changed history");
    }

    if (disabled.result_ids != disabled.selected) {
        throw std::runtime_error(
                test_case.name + " selector/result token ownership mismatch");
    }
    std::vector<std::size_t> expected_history_after;
    if (test_case.limit != 0) {
        expected_history_after = test_case.prompt;
        expected_history_after.insert(
                expected_history_after.end(), disabled.result_ids.begin(),
                disabled.result_ids.end());
    }
    if (disabled.history_after != expected_history_after) {
        throw std::runtime_error(
                test_case.name + " committed history relationship mismatch");
    }
    const std::size_t observed_kv = test_case.limit == 0
            ? 0
            : (disabled.result_ids.empty()
                       ? test_case.prompt.size()
                       : test_case.prompt.size()
                               + disabled.result_ids.size() - 1);
    if (disabled.cache_lengths.size() != test_case.layers) {
        throw std::runtime_error(
                test_case.name
                + " decoder-layer cache cardinality mismatch");
    }
    for (const std::size_t length : disabled.cache_lengths) {
        if (length != observed_kv) {
            throw std::runtime_error(
                    test_case.name
                    + " terminal token/cache relationship mismatch");
        }
    }
    const std::string observed_stop = stop_name(disabled.stop_reason);
    const std::string internally_expected_stop = [&] {
        if (test_case.limit == 0) return std::string("max_new_tokens");
        if (disabled.result_ids.empty()) {
            return std::string("context_exhaustion");
        }
        if (disabled.result_ids.back() == 2) return std::string("eos");
        if (disabled.result_ids.size() >= test_case.limit) {
            return std::string("max_new_tokens");
        }
        return std::string("context_exhaustion");
    }();
    if (observed_stop != internally_expected_stop) {
        throw std::runtime_error(
                test_case.name + " stop precedence relationship mismatch");
    }

    const auto& expected = test_case.record.at("expected_result");
    const std::string expected_context =
            test_case.name + ".expected_result";
    if (same_history) {
        const std::vector<std::size_t> reference_result = token_ids(
                expected.at("token_ids"), expected_context + ".token_ids");
        if (reference_result != disabled.result_ids
                || string_member(expected, "stop_reason", expected_context)
                        != observed_stop) {
            throw std::runtime_error(
                    test_case.name + " comparable reference result mismatch");
        }
        if (size_member(
                    expected, "initialized_kv_length", expected_context)
                != observed_kv) {
            throw std::runtime_error(
                    test_case.name + " comparable reference KV mismatch");
        }
    }

    if (disabled.request_length != test_case.prompt.size()
            || disabled.poisoned) {
        throw std::runtime_error(
                test_case.name + " request ownership state mismatch");
    }
    if (!metrics.snapshot.has_value() || !traced.snapshot.has_value()) {
        throw std::runtime_error(
                test_case.name + " missing metrics snapshot");
    }
    compare_snapshots(*metrics.snapshot, *traced.snapshot, test_case.name);
    const iom::InferenceSnapshot& scalar = *metrics.snapshot;
    if (!scalar.request_admitted
            || scalar.attempt.outcome != iom::AttemptOutcome::succeeded
            || scalar.admitted.prompt_tokens != test_case.prompt.size()
            || scalar.admitted.generated_tokens
                    != disabled.result_ids.size()
            || scalar.admitted.decode_forward_count
                    != (disabled.result_ids.empty()
                                ? 0
                                : disabled.result_ids.size() - 1)
            || scalar.admitted.decode_token_count
                    != (disabled.result_ids.empty()
                                ? 0
                                : disabled.result_ids.size() - 1)
            || !scalar.admitted.stop_reason_valid
            || stop_name(scalar.admitted.stop_reason) != observed_stop) {
        throw std::runtime_error(
                test_case.name + " scalar metric relationship mismatch");
    }
    if (!metrics.operations.empty() || metrics.trace_prepared
            || !traced.trace_prepared || traced.trace_rows_dropped != 0
            || traced.operations.size() != traced.accepted_operations) {
        throw std::runtime_error(
                test_case.name + " instrumentation mode state mismatch");
    }
    if (test_case.limit == 0) {
        if (!disabled.logits.empty() || !disabled.selected.empty()
                || !disabled.result_ids.empty()
                || !disabled.histories.empty()
                || !disabled.history_after.empty()
                || disabled.accepted_operations != 0
                || !traced.operations.empty()
                || scalar.admitted.prefill.state
                        != iom::ObservationState::not_run
                || scalar.admitted.decode.state
                        != iom::ObservationState::not_run
                || scalar.admitted.time_to_first_token_valid
                || scalar.admitted.time_to_first_token_ns != 0
                || scalar.admitted.decode_throughput_valid
                || scalar.admitted.decode_throughput_denominator_ns != 0
                || scalar.admitted.tokens_per_second != 0.0) {
            throw std::runtime_error(
                    test_case.name
                    + " did work or invented a zero-limit timing rate");
        }
    } else if (traced.operations.empty()) {
        throw std::runtime_error(
                test_case.name + " trace rows are missing");
    }

    nlohmann::json trace_records = nlohmann::json::array();
    std::unordered_set<iom::oid> operation_ids;
    const std::size_t decode_forwards =
            disabled.result_ids.empty()
            ? 0
            : disabled.result_ids.size() - 1;
    if (test_case.layers == 0) {
        throw std::runtime_error(
                test_case.name + " has no decoder layers to attribute");
    }
    std::set<std::size_t> decode_positions;
    bool saw_whole_prefill = false;
    bool saw_terminal_prefill = false;
    for (const auto& row : traced.operations) {
        if (!operation_ids.insert(row.operation).second || row.operation <= 0
                || row.request_ordinal != traced.snapshot->admitted.ordinal
                || row.run_length == 0
                || (row.decoder_layer
                    && *row.decoder_layer >= test_case.layers)
                || row.wait_state != iom::WaitState::succeeded
                || !row.wait_observed_ns.has_value()) {
            throw std::runtime_error(
                    test_case.name + " trace attribution is invalid");
        }
        if (row.phase == iom::InferencePhase::prefill) {
            const bool whole_prompt = row.position_start == 0
                    && row.run_length == test_case.prompt.size();
            const bool terminal_window = !row.decoder_layer
                    && row.run_length == 1
                    && row.position_start + row.run_length
                            == test_case.prompt.size();
            if (!whole_prompt && !terminal_window) {
                throw std::runtime_error(
                        test_case.name
                        + " prefill absolute-position attribution is invalid");
            }
            saw_whole_prefill = saw_whole_prefill || whole_prompt;
            saw_terminal_prefill = saw_terminal_prefill || terminal_window;
        } else if (row.phase == iom::InferencePhase::decode) {
            if (row.run_length != 1 || decode_forwards == 0
                    || row.position_start < test_case.prompt.size()
                    || row.position_start
                            >= test_case.prompt.size() + decode_forwards) {
                throw std::runtime_error(
                        test_case.name
                        + " decode absolute-position attribution is invalid");
            }
            decode_positions.insert(row.position_start);
        } else {
            throw std::runtime_error(
                    test_case.name + " trace phase is invalid");
        }
        trace_records.push_back({
                {"request_ordinal", row.request_ordinal},
                {"oid", row.operation},
                {"phase", row.phase == iom::InferencePhase::prefill
                                  ? "prefill"
                                  : "decode"},
                {"decoder_layer",
                 row.decoder_layer
                         ? nlohmann::json(*row.decoder_layer)
                         : nlohmann::json(nullptr)},
                {"position_start", row.position_start},
                {"run_length", row.run_length},
                {"wait_state", "succeeded"}});
    }
    if (test_case.limit != 0) {
        if (!saw_whole_prefill || !saw_terminal_prefill) {
            throw std::runtime_error(
                    test_case.name
                    + " trace is missing whole-prompt or terminal prefill "
                      "attribution");
        }
        for (std::size_t index = 0; index < decode_forwards; ++index) {
            if (!decode_positions.contains(test_case.prompt.size() + index)) {
                throw std::runtime_error(
                        test_case.name
                        + " trace is missing a cached-decode position");
            }
        }
    }

    evidence = {
            {"id", test_case.name},
            {"selection_policy", test_case.policy},
            {"reference_history_matched", same_history},
            {"logical_run",
             {{"prompt_ids", test_case.prompt},
              {"prompt_positions",
               test_case.record.at("input").at("positions")},
              {"selector_histories", disabled.histories},
              {"selected_ids", disabled.selected}}},
            {"result",
             {{"stop_reason", observed_stop},
              {"token_ids", disabled.result_ids},
              {"decoded_text", disabled.text},
              {"request_length", disabled.request_length},
              {"cache_lengths", disabled.cache_lengths},
              {"history", disabled.history_after},
              {"poisoned", disabled.poisoned}}},
            {"logits", disabled.logits},
            {"metrics",
             {{"prompt_tokens", scalar.admitted.prompt_tokens},
              {"generated_tokens", scalar.admitted.generated_tokens},
              {"decode_forward_count",
               scalar.admitted.decode_forward_count},
              {"decode_token_count", scalar.admitted.decode_token_count},
              {"time_to_first_token_available",
               scalar.admitted.time_to_first_token_valid},
              {"time_to_first_token_ns",
               scalar.admitted.time_to_first_token_ns},
              {"decode_rate_available",
               scalar.admitted.decode_throughput_valid},
              {"decode_rate_denominator_ns",
               scalar.admitted.decode_throughput_denominator_ns},
              {"tokens_per_second", scalar.admitted.tokens_per_second}}},
            {"trace",
             {{"prepared", traced.trace_prepared},
              {"accepted_operations", traced.accepted_operations},
              {"rows_dropped", traced.trace_rows_dropped},
              {"rows", std::move(trace_records)}}},
            {"mode_parity", true},
            {"native_capability", "unclaimed"}};
}

inline void write_evidence(
        const std::filesystem::path& path, const nlohmann::json& value) {
    if (path.empty()) throw std::invalid_argument("IOM_TEST_MODEL_EVIDENCE must not be empty");
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::is_directory(parent)) throw std::runtime_error("official evidence parent does not exist: " + parent.string());
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open official evidence temporary file: " + temporary.string());
        output << value.dump(2) << '\n';
        if (!output) throw std::runtime_error("cannot write official evidence: " + path.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
        if (error) throw std::runtime_error("cannot publish official evidence: " + path.string());
    }
}

inline void run_real_model_inference_impl(
        iom::Device& device, const std::filesystem::path& model_directory,
        const std::string& artifact_id,
        const std::filesystem::path& reference_path,
        const std::filesystem::path& evidence_path,
        EvidencePublication& publication,
        nlohmann::json& evidence) {
    const std::size_t reference_size = official_file_size(reference_path);
    const std::string reference_sha256 = sha256_file(reference_path);
    const nlohmann::json pack = read_json(reference_path, "official reference pack");
    const std::vector<OfficialCase> cases =
            validate_reference_pack(pack, artifact_id);
    const nlohmann::json& provenance = pack.at("provenance");

    // The path-based loader cannot retain caller file handles.  Revalidate the
    // complete size/SHA-256 inventory immediately before every independent
    // config/session load and once more before evidence publication.  A caller
    // must keep the directory immutable for the run; any observed replacement
    // or in-place mutation fails rather than binding results to stale hashes.
    validate_artifacts(model_directory, provenance);
    const iom::TinyLlamaConfig config =
            iom::load_tinyllama_config(model_directory);
    validate_model_geometry(config, pack.at("model_config"));
    if (const auto arena =
                optional_environment("IOM_TEST_MODEL_ARENA_BYTES")) {
        static_cast<void>(parse_arena_bytes(*arena));
    }

    evidence["reference"] = {
            {"path", reference_path.string()},
            {"size", reference_size},
            {"sha256", reference_sha256},
            {"artifact_id", artifact_id},
            {"case_payload_sha256", provenance.at("case_payload_sha256")},
            {"artifacts", provenance.at("artifacts")}};
    evidence["model"] = {
            {"directory", model_directory.string()},
            {"model_config", pack.at("model_config")},
            {"immutability",
             "size and SHA-256 inventory revalidated at every load boundary "
             "and before publication"}};
    evidence["cases"] = nlohmann::json::array();
    for (const OfficialCase& test_case : cases) {
        std::array<ModeObservation, 3> observations;
        for (std::size_t mode = 0; mode < observations.size(); ++mode) {
            validate_artifacts(model_directory, provenance);
            observations[mode] = run_mode(
                    device, model_directory, test_case,
                    static_cast<int>(mode));
        }
        nlohmann::json case_evidence;
        compare_case(test_case, observations, case_evidence);
        evidence["cases"].push_back(std::move(case_evidence));
    }
    validate_artifacts(model_directory, provenance);
    if (official_file_size(reference_path) != reference_size
            || sha256_file(reference_path) != reference_sha256) {
        throw std::runtime_error(
                "official reference identity changed during execution");
    }
    publication.validate();
    evidence["status"] = "passed";
    write_evidence(evidence_path, evidence);
}

}  // namespace official_model_detail

// Called unchanged by each opt-in backend driver.  Every caller input is
// explicit; ordinary conformance never enters this function.
inline void run_real_model_inference(iom::Device& device) {
    const std::string model_directory_value = official_model_detail::required_environment("IOM_TEST_MODEL_DIR");
    const std::string artifact_id = official_model_detail::required_environment("IOM_TEST_MODEL_ID");
    const std::string reference_value = official_model_detail::required_environment("IOM_TEST_MODEL_REFERENCE");
    const std::string evidence_value = official_model_detail::required_environment("IOM_TEST_MODEL_EVIDENCE");
    const std::filesystem::path model_directory(model_directory_value);
    const std::filesystem::path reference_path(reference_value);
    const std::filesystem::path evidence_path(evidence_value);
    official_model_detail::EvidencePublication publication(
            evidence_path, model_directory, reference_path);
    publication.clear_stale();
    nlohmann::json environment = {
            {"IOM_TEST_MODEL_DIR", model_directory_value},
            {"IOM_TEST_MODEL_ID", artifact_id},
            {"IOM_TEST_MODEL_REFERENCE", reference_value},
            {"IOM_TEST_MODEL_EVIDENCE", evidence_value}};
    if (const char* arena = std::getenv("IOM_TEST_MODEL_ARENA_BYTES")) {
        environment["IOM_TEST_MODEL_ARENA_BYTES"] = arena;
    }
    nlohmann::json evidence = {
            {"schema_version", 1},
            {"kind", "official-inference-evidence"},
            {"status", "failed"},
            {"argv", official_model_detail::invocation_record()},
            {"backend",
             official_model_detail::backend_name(device.backend_kind())},
            {"device", device.backend_device()},
            {"device_identity",
             [&] {
                 std::ostringstream output;
                 output << static_cast<const void*>(&device);
                 return output.str();
             }()},
            {"software",
             {{"component", "iom official inference harness"},
              {"cplusplus", __cplusplus}}},
            {"precision",
             {{"weights", "BF16"},
              {"activations", "BF16"},
              {"logits", "BF16"},
              {"accumulation", "FP32"}}},
            {"native_capability", "unclaimed"},
            {"environment", std::move(environment)}};
    try {
        if (!std::filesystem::is_directory(model_directory)) {
            throw std::invalid_argument(
                    "IOM_TEST_MODEL_DIR must name an existing directory");
        }
        official_model_detail::require_regular_nonsymlink(
                reference_path, "IOM_TEST_MODEL_REFERENCE");
        official_model_detail::run_real_model_inference_impl(
                device, model_directory, artifact_id, reference_path,
                evidence_path, publication, evidence);
    } catch (...) {
        evidence["status"] = "failed";
        evidence["failure"] =
                official_model_detail::failure_message(
                        std::current_exception());
        evidence["diagnostics"] =
                nlohmann::json::array({evidence["failure"]});
        try {
            publication.validate();
            official_model_detail::write_evidence(evidence_path, evidence);
        } catch (...) { }
        throw;
    }
}

}  // namespace iom_conformance
