#pragma once

// Backend-neutral physical-storage oracle for the conformance harness.
// The encoder deliberately depends only on the public tensor metadata and the
// checked standard-layout helper; backend drivers provide the native access.

#include "backend/backend_conformance_common.hpp"

#include <functional>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <span>
#include <utility>
#include <vector>

namespace iom_conformance {

class AcceleratorStorageOracle {
public:
    virtual ~AcceleratorStorageOracle() = default;

    // The harness supplies the owner specification before operating on any
    // view. This is non-virtual so every oracle has the same owner contract.
    void set_owner_spec(const iom::TensorSpec& spec) noexcept {
        owner_spec_ = &spec;
    }

    virtual void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) = 0;
    [[nodiscard]] virtual std::vector<std::byte> observe(
            const iom::TensorView& view) const = 0;

protected:
    [[nodiscard]] const iom::TensorSpec& owner_spec() const {
        if (owner_spec_ == nullptr) {
            throw std::logic_error("storage oracle owner specification is unset");
        }
        return *owner_spec_;
    }

private:
    const iom::TensorSpec* owner_spec_ = nullptr;
};

// Read a packed element from a storage buffer. The bit order matches the
// independent host encoder in backend_conformance_common.hpp.
inline std::uint64_t read_storage_bits(
        const std::byte* base, std::size_t bit_offset, std::size_t nbits) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(base);
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < nbits; ++i) {
        if ((bytes[(bit_offset + i) / 8] >> ((bit_offset + i) % 8)) & 1u) {
            value |= std::uint64_t{1} << i;
        }
    }
    return value;
}

inline void write_storage_bits(
        std::byte* base, std::size_t bit_offset, std::size_t nbits,
        std::uint64_t value) {
    write_bits(
            reinterpret_cast<unsigned char*>(base), bit_offset, nbits, value);
}

// Owner storage slot for the linear-th element of a transformed view. This
// maps only leading planes; final row and column coordinates stay unchanged.
// Tile arithmetic remains exclusively in iom::detail::standard_layout_slot.
inline std::size_t standard_layout_view_slot(
        const iom::TensorView& view, const iom::TensorSpec& owner,
        std::size_t linear) {
    const std::span<const std::size_t> view_dims = view.spec().shape.dimensions();
    const std::size_t view_leading_rank = view_dims.size() - 2;
    const std::size_t rows = view_dims[view_leading_rank];
    const std::size_t columns = view_dims[view_leading_rank + 1];
    const std::span<const std::size_t> view_strides = view.plane_strides();
    const std::span<const std::size_t> owner_dims = owner.shape.dimensions();
    const std::size_t owner_leading_rank = owner_dims.size() - 2;

    std::size_t rest = linear;
    const std::size_t column = rest % columns;
    rest /= columns;
    const std::size_t row = rest % rows;
    rest /= rows;

    std::size_t plane = view.plane_offset();
    for (std::size_t axis = view_leading_rank; axis-- > 0;) {
        plane += (rest % view_dims[axis]) * view_strides[axis];
        rest /= view_dims[axis];
    }

    std::vector<std::size_t> coordinates(owner_dims.size());
    for (std::size_t axis = owner_leading_rank; axis-- > 0;) {
        coordinates[axis] = plane % owner_dims[axis];
        plane /= owner_dims[axis];
    }
    coordinates[owner_leading_rank] = row;
    coordinates[owner_leading_rank + 1] = column;
    return iom::detail::standard_layout_slot(
            owner, std::span<const std::size_t>{coordinates});
}
// Encode the standard 16x16 tiled allocation for the deterministic logical
// pattern at salt zero. Padded coordinates are visited in row-major order;
// padding remains zero and every in-bounds coordinate obtains its slot from
// the checked core helper rather than duplicating tile math in the test.
inline std::vector<std::byte> encode_standard_tiled_storage(
        const iom::TensorSpec& spec) {
    spec.validate();
    const std::size_t bits = iom::detail::leaf_bits(spec.data_type);
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    const iom::TensorShape padded_shape = spec.standard_padded_shape();
    const std::span<const std::size_t> padded = padded_shape.dimensions();
    const std::size_t count = spec.shape.element_count();
    const std::size_t padded_count = padded_shape.element_count();
    std::vector<std::byte> storage(spec.tiled_storage_nbytes(), std::byte{0});

    std::vector<std::size_t> coordinates(dimensions.size());
    for (std::size_t padded_linear = 0; padded_linear < padded_count;
         ++padded_linear) {
        std::size_t rest = padded_linear;
        bool in_bounds = true;
        for (std::size_t axis = padded.size(); axis-- > 0;) {
            coordinates[axis] = rest % padded[axis];
            rest /= padded[axis];
            in_bounds = in_bounds && coordinates[axis] < dimensions[axis];
        }
        if (!in_bounds) {
            continue;
        }

        std::size_t logical_linear = 0;
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            logical_linear = logical_linear * dimensions[axis] + coordinates[axis];
        }
        if (logical_linear >= count) {
            throw std::logic_error("standard storage encoder logical index overflow");
        }
        const std::size_t slot = iom::detail::standard_layout_slot(
                spec, std::span<const std::size_t>{coordinates});
        write_storage_bits(
                storage.data(), slot * bits, bits,
                element_pattern(spec.data_type, logical_linear, 0));
    }
    return storage;
}

inline std::vector<std::byte> decode_standard_tiled_view(
        const iom::TensorView& view, const iom::TensorSpec& owner,
        std::span<const std::byte> storage) {
    if (storage.size() != owner.tiled_storage_nbytes()) {
        throw std::invalid_argument("standard storage decode has the wrong size");
    }
    const std::size_t bits = iom::detail::leaf_bits(view.spec().data_type);
    const std::size_t count = view.spec().shape.element_count();
    std::vector<std::byte> logical(view.spec().logical_nbytes(), std::byte{0});
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::size_t slot = standard_layout_view_slot(view, owner, linear);
        write_storage_bits(
                logical.data(), linear * bits, bits,
                read_storage_bits(storage.data(), slot * bits, bits));
    }
    return logical;
}

inline void apply_standard_tiled_view(
        const iom::TensorView& view, const iom::TensorSpec& owner,
        std::span<const std::byte> logical, std::vector<std::byte>& storage) {
    if (logical.size() != view.spec().logical_nbytes()
            || storage.size() != owner.tiled_storage_nbytes()) {
        throw std::invalid_argument("standard storage view update has the wrong size");
    }
    const std::size_t bits = iom::detail::leaf_bits(view.spec().data_type);
    const std::size_t count = view.spec().shape.element_count();
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::size_t slot = standard_layout_view_slot(view, owner, linear);
        const std::uint64_t value = read_storage_bits(
                logical.data(), linear * bits, bits);
        write_storage_bits(storage.data(), slot * bits, bits, value);
    }
}

// CPU's allocation is already the standard layout, so it is useful both as a
// local oracle implementation and as a compile-only reference for the common
// encoder.
class CpuStorageOracle final : public AcceleratorStorageOracle {
public:
    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        require_size(view, encoded.size());
        std::memcpy(view.native_handle(), encoded.data(), encoded.size());
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        std::memcpy(result.data(), view.native_handle(), bytes);
        return result;
    }

private:
    void require_size(const iom::TensorView&, std::size_t actual) const {
        if (actual != owner_spec().tiled_storage_nbytes()) {
            throw std::invalid_argument("CPU storage oracle received the wrong size");
        }
    }
};

// Deliberately perturb a logical-to-physical slot map in both directions.
// Used only by negative conformance fixtures to prove that a round trip
// through a matching wrong map can cancel while the oracle detects it.
class PermutingStorageOracle final : public AcceleratorStorageOracle {
public:
    using SlotMap = std::function<std::size_t(std::size_t)>;

    PermutingStorageOracle(
            AcceleratorStorageOracle& delegate, SlotMap map)
            : delegate_(delegate), map_(std::move(map)) {}

    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        delegate_.set_owner_spec(owner_spec());
        std::vector<std::byte> perturbed = permute(encoded);
        delegate_.seed(view, perturbed);
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        delegate_.set_owner_spec(owner_spec());
        std::vector<std::byte> observed = delegate_.observe(view);
        return permute(observed);
    }

private:
    [[nodiscard]] std::vector<std::byte> permute(
            std::span<const std::byte> storage) const {
        const std::size_t bits = iom::detail::leaf_bits(owner_spec().data_type);
        const std::size_t slots =
                owner_spec().standard_padded_shape().element_count();
        if (storage.size() != owner_spec().tiled_storage_nbytes()) {
            throw std::invalid_argument(
                    "permuting storage oracle received the wrong size");
        }
        std::vector<std::byte> result(storage.size(), std::byte{0});
        for (std::size_t logical = 0; logical < slots; ++logical) {
            const std::size_t physical = map_(logical);
            if (physical >= slots) {
                throw std::out_of_range(
                        "permuting storage oracle map exceeds allocation");
            }
            write_storage_bits(
                    result.data(), physical * bits, bits,
                    read_storage_bits(storage.data(), logical * bits, bits));
        }
        return result;
    }

    AcceleratorStorageOracle& delegate_;
    SlotMap map_;
};

inline std::size_t swap_first_adjacent_slots(std::size_t logical) {
    if (logical == 0) {
        return 1;
    }
    if (logical == 1) {
        return 0;
    }
    return logical;
}

}  // namespace iom_conformance
