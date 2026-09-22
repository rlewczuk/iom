#pragma once

// Shared synthetic TinyLlama fixture for library-level inference observation
// checks.  The checkpoint and tokenizer writers are the existing model-loading
// and tokenizer fixtures; this header only supplies the finite BF16 payload,
// tokenizer metadata required by the session factory, and the deterministic
// caller-owned selector used by both the focused session tests and the
// backend-neutral parity scenario.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/oid.hpp"
#include "iom/tensor.hpp"
#include "iom/token_selection.hpp"
#include "model_loading_fixture.hpp"
#include "tokenizer_fixture.hpp"

namespace iom_inference_metrics_test {

using iom_model_loading::required_weight_entries;
using iom_model_loading::h16_config;
using iom_model_loading::one_layer_config;
using iom_model_loading::write_config;
using iom_model_loading::write_file;
using iom_tokenizer_test::tokenizer_config;
using iom_tokenizer_test::write_tokenizer;

[[nodiscard]] inline std::uint16_t forward_encode_bf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16);
}

[[nodiscard]] inline float forward_decode_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// Finite, small positive values keep every backend's native BF16 path in the
// same ordinary numerical class while still making every checkpoint entry
// distinct.  The session selector owns token expectations; logits are compared
// only between disabled and enabled runs of the same backend.
[[nodiscard]] inline std::string forward_finite_payload(
        const std::vector<std::size_t>& shape, std::size_t entry) {
    const std::size_t elements = iom_model_loading::element_count(shape);
    std::string payload(elements * sizeof(std::uint16_t), '\0');
    for (std::size_t element = 0; element < elements; ++element) {
        float value = 0.015F
                * static_cast<float>(1 + ((entry + element) % 7));
        if (entry == 1
                || (entry >= 3 && ((entry - 3) % 9) < 2)) {
            value = 1.0F + 0.015F
                    * static_cast<float>((entry + element) % 3);
        }
        const std::uint16_t bits = forward_encode_bf16(value);
        std::memcpy(payload.data() + element * sizeof(bits), &bits,
                    sizeof(bits));
    }
    return payload;
}

inline void write_forward_weights(
        const std::filesystem::path& directory,
        const nlohmann::json& config) {
    auto entries = required_weight_entries(config);
    for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        entries[entry].payload =
                forward_finite_payload(entries[entry].shape, entry);
    }
    write_safetensors_file(directory, "model.safetensors", entries);
}

// Directory-only ownership keeps backend setup in the driver and lets the
// same bytes feed CPU, CUDA, ROCm, and SYCL sessions without a second fixture
// framework or a backend branch in this header.
struct SyntheticFixture {
    iom_model_loading::TempDir directory;

    explicit SyntheticFixture(
            std::string tag,
            const nlohmann::json& config = one_layer_config())
        : directory("inference-metrics-" + std::move(tag)) {
        write_config(directory.path(), config);
        write_forward_weights(directory.path(), config);
        write_tokenizer(directory.path());
        write_file(directory.path(), "tokenizer_config.json",
                   tokenizer_config().dump());
    }
};

class SequenceSelector final : public iom::TokenSelector {
public:
    explicit SequenceSelector(std::vector<std::size_t> sequence)
        : sequence_(std::move(sequence)) {}

    [[nodiscard]] iom::TokenSelectorScratchRequirements scratch_requirements(
            const iom::TensorView&, std::size_t) const override {
        return {0, {0, 1}};
    }

    [[nodiscard]] std::size_t select(
            iom::DeviceOps&, const iom::TensorView&, std::size_t,
            iom::oid producer, std::span<const std::size_t> history,
            iom::TokenSelectorScratch) override {
        if (before_select) {
            before_select(calls);
        }
        ++calls;
        producers.push_back(producer);
        histories.emplace_back(history.begin(), history.end());
        if (throw_failure) {
            throw std::runtime_error("injected selector failure");
        }
        if (next >= sequence_.size()) {
            throw std::logic_error("injected selector sequence exhausted");
        }
        return sequence_[next++];
    }

    std::size_t calls = 0;
    std::vector<iom::oid> producers;
    std::vector<std::vector<std::size_t>> histories;
    bool throw_failure = false;
    std::function<void(std::size_t)> before_select;

private:
    std::vector<std::size_t> sequence_;
    std::size_t next = 0;
};

}  // namespace iom_inference_metrics_test
