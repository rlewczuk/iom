#include "iom/token_selection.hpp"

#include "iom/iom.hpp"
#include "iom_internal.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace iom {
namespace {

constexpr const char* kSelectorOperation = "greedy token selection";

[[nodiscard]] std::size_t validate_logits_shape(
        const TensorView& logits, std::size_t valid_vocabulary) {
    const TensorSpec& spec = logits.spec();
    if (spec.data_type != DataType::BF16) {
        throw std::invalid_argument(
                "greedy token selection requires BF16 logits");
    }
    if (spec.quantization != QuantizationFormat::NONE) {
        throw std::invalid_argument(
                "greedy token selection requires unquantized logits");
    }

    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    if (spec.shape.rank() != 2 || dimensions.size() != 2
            || dimensions[0] != 1 || dimensions[1] == 0) {
        throw std::invalid_argument(
                "greedy token selection requires logits with shape [1,V]");
    }
    if (valid_vocabulary == 0 || valid_vocabulary != dimensions[1]) {
        throw std::invalid_argument(
                "valid vocabulary must equal the logits vocabulary");
    }
    return detail::checked_mul(
            valid_vocabulary, sizeof(std::uint16_t),
            "greedy token selection host byte count overflows");
}

[[nodiscard]] detail::CheckedViewFacts validate_logits_view(
        const Device& device, const TensorView& logits,
        std::size_t valid_vocabulary) {
    const std::size_t host_bytes =
            validate_logits_shape(logits, valid_vocabulary);
    const Tensor* owner = logits.owner_identity();
    if (owner == nullptr) {
        throw std::invalid_argument(
                "greedy token selection logits have no owner");
    }
    const detail::CheckedViewFacts facts = detail::validate_checked_view(
            device, logits, kSelectorOperation);
    if (facts.logical_bytes != host_bytes) {
        throw std::invalid_argument(
                "greedy token selection logits have inconsistent byte extent");
    }
    return facts;
}

void validate_history(std::span<const std::size_t> history) {
    if (!history.empty() && history.data() == nullptr) {
        throw std::invalid_argument(
                "greedy token selection history has no storage");
    }
}

void validate_host_scratch(
        std::span<std::byte> host, std::size_t required_bytes,
        const TensorView& logits, std::size_t owner_storage_bytes) {
    if (host.size() < required_bytes) {
        throw std::invalid_argument(
                "greedy token selection host scratch is too small");
    }
    if (!host.empty() && host.data() == nullptr) {
        throw std::invalid_argument(
                "greedy token selection host scratch has no storage");
    }

    const auto* owner_handle = logits.native_handle();
    if (owner_handle == nullptr) {
        throw std::invalid_argument(
                "greedy token selection logits have no native storage");
    }
    if (host.empty()) return;

    const std::uintptr_t host_begin =
            reinterpret_cast<std::uintptr_t>(host.data());
    const std::uintptr_t owner_begin =
            reinterpret_cast<std::uintptr_t>(owner_handle);
    const std::uintptr_t address_limit =
            std::numeric_limits<std::uintptr_t>::max();
    if (host.size() > address_limit
            || owner_storage_bytes > address_limit) {
        throw std::overflow_error(
                "greedy token selection scratch range overflows");
    }
    const std::uintptr_t host_size = static_cast<std::uintptr_t>(host.size());
    const std::uintptr_t owner_size =
            static_cast<std::uintptr_t>(owner_storage_bytes);
    if (host_size > address_limit - host_begin
            || owner_size > address_limit - owner_begin) {
        throw std::overflow_error(
                "greedy token selection scratch range overflows");
    }
    const std::uintptr_t host_end = host_begin + host_size;
    const std::uintptr_t owner_end = owner_begin + owner_size;
    if (host_begin < owner_end && owner_begin < host_end) {
        throw std::invalid_argument(
                "greedy token selection host scratch overlaps logits");
    }
}

void validate_device_scratch(
        const Device& device, const TensorView& logits,
        const RawWorkspaceView& scratch,
        const WorkspaceRequirements& requirements) {
    const std::span<const TensorView> operands(&logits, 1);
    static_cast<void>(detail::WorkspaceValidation::validated(
            device, scratch, requirements.bytes, requirements.alignment,
            operands));
}

[[nodiscard]] std::uint16_t load_bf16(
        std::span<const std::byte> bytes, std::size_t index) noexcept {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(bytes.data());
    const std::size_t offset = index * sizeof(std::uint16_t);
    return static_cast<std::uint16_t>(raw[offset])
            | static_cast<std::uint16_t>(raw[offset + 1]) << 8;
}

[[nodiscard]] float decode_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

[[nodiscard]] std::size_t scan_logits(
        std::span<const std::byte> bytes, std::size_t vocabulary) {
    const std::uint16_t first_bits = load_bf16(bytes, 0);
    if ((first_bits & 0x7F80u) == 0x7F80u) {
        throw std::runtime_error(
                "greedy token selection encountered a nonfinite logit");
    }
    float best = decode_bf16(first_bits);
    std::size_t winner = 0;
    for (std::size_t id = 1; id < vocabulary; ++id) {
        const std::uint16_t bits = load_bf16(bytes, id);
        if ((bits & 0x7F80u) == 0x7F80u) {
            throw std::runtime_error(
                    "greedy token selection encountered a nonfinite logit");
        }
        const float value = decode_bf16(bits);
        if (value > best) {
            best = value;
            winner = id;
        }
    }
    return winner;
}

}  // namespace

TokenSelectorScratchRequirements GreedyTokenSelector::scratch_requirements(
        const TensorView& logits, std::size_t valid_vocabulary) const {
    const Tensor* owner = logits.owner_identity();
    if (owner == nullptr) {
        throw std::invalid_argument(
                "greedy token selection logits have no owner");
    }
    const std::size_t host_bytes =
            validate_logits_shape(logits, valid_vocabulary);
    const Device& device = logits.device();
    const detail::CheckedViewFacts facts = detail::validate_checked_view(
            device, logits, kSelectorOperation);
    if (facts.logical_bytes != host_bytes) {
        throw std::invalid_argument(
                "greedy token selection logits have inconsistent byte extent");
    }
    return TokenSelectorScratchRequirements{
            host_bytes, logits.copy_to_host_workspace_requirements()};
}

std::size_t GreedyTokenSelector::select(
        DeviceOps& queue, const TensorView& logits,
        std::size_t valid_vocabulary, oid producer,
        std::span<const std::size_t> history, TokenSelectorScratch scratch) {
    const Device* queue_device = nullptr;
    try {
        queue_device = &queue.device();
    } catch (const std::logic_error&) {
        throw std::invalid_argument(
                "greedy token selection queue has no device");
    }

    if (producer <= 0) {
        throw std::invalid_argument(
                "greedy token selection producer OID must be positive");
    }
    validate_history(history);

    const std::size_t host_bytes =
            validate_logits_shape(logits, valid_vocabulary);
    const detail::CheckedViewFacts facts = validate_logits_view(
            *queue_device, logits, valid_vocabulary);
    const WorkspaceRequirements device_requirements =
            logits.copy_to_host_workspace_requirements();
    if (facts.logical_bytes != host_bytes) {
        throw std::invalid_argument(
                "greedy token selection logits have inconsistent byte extent");
    }
    validate_host_scratch(
            scratch.host, host_bytes, logits, facts.storage_bytes);
    validate_device_scratch(
            *queue_device, logits, scratch.device, device_requirements);

    queue.wait(producer);
    const std::span<std::byte> transfer_destination =
            scratch.host.first(host_bytes);
    logits.copy_to_host(transfer_destination, scratch.device);
    return scan_logits(
            std::span<const std::byte>(
                    transfer_destination.data(), transfer_destination.size()),
            valid_vocabulary);
}

}  // namespace iom
