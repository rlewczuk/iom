#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <span>

#include "iom/detail/aligned_storage.hpp"
#include "iom/iom.hpp"

namespace iom {
namespace {

class CpuStorageCleanup final : public detail::CleanupAction {
public:
    CpuStorageCleanup(Allocator& allocator, void* address)
        : allocator_(allocator), address_(address) {}
    void run() noexcept override {
        detail::release_aligned_storage(allocator_, address_);
        attempted_ = true;
    }
    [[nodiscard]] bool completed() const noexcept override { return attempted_; }
    [[nodiscard]] bool failed() const noexcept override { return false; }
    [[nodiscard]] std::exception_ptr failure() const noexcept override {
        return nullptr;
    }
private:
    Allocator& allocator_;
    void* address_;
    bool attempted_ = false;
};

class CpuTensor final : public Tensor {
public:
    CpuTensor(const TensorSpec& spec, CpuDevice& device, Allocator& allocator)
            : Tensor(spec, device), device_(device), allocator_(allocator) {
        address_ = detail::allocate_aligned_storage(
                allocator_, view().spec().tiled_storage_nbytes(), [] {},
                "CPU tensor storage is not 32-byte aligned");
        try {
            std::fill_n(static_cast<std::byte*>(address_),
                        view().spec().tiled_storage_nbytes(), std::byte{0});
        } catch (...) {
            detail::release_aligned_storage(allocator_, address_);
            throw;
        }
    }

    ~CpuTensor() noexcept override {
        void* address = address_;
        if (address == nullptr) {
            return;
        }
        detail::release_or_quarantine(
                device_.registry_state().registry, address,
                [this, address] {
                    try {
                        device_.registry_state().quarantine.emplace<
                                CpuStorageCleanup>(allocator_, address);
                        address_ = nullptr;
                    } catch (...) {
                    }
                },
                [this] {
                    detail::release_aligned_storage(allocator_, address_);
                });
    }

private:
    [[nodiscard]] void* storage_handle() noexcept override { return address_; }

    void region_from_host(
            const TensorView& destination, std::span<const std::byte> source,
            RawWorkspaceView) override {
        auto* storage = static_cast<unsigned char*>(address_);
        const auto* host = reinterpret_cast<const unsigned char*>(source.data());
        const std::size_t bits = detail::leaf_bits(destination.spec().data_type);
        const std::size_t columns =
                destination.spec().shape.dimensions().back();
        const auto dimensions = destination.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t tile_columns =
                columns / TensorSpec::TILE + (columns % TensorSpec::TILE != 0);
        const std::size_t callbacks_per_plane = rows * tile_columns;
        const std::size_t elements_per_plane = rows * columns;
        std::size_t callback_index = 0;
        std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
        std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
        if (bits % 8 != 0) {
            for (std::size_t index = 0; index < TensorSpec::TILE; ++index) {
                shift_table[index] =
                        (index * bits) % (sizeof(std::uint32_t) * 8);
                mask_table[index] =
                        ((1u << bits) - 1u) << shift_table[index];
            }
        }
        cpu_detail::for_each_tile(
                destination,
                [&](std::size_t row, std::size_t column,
                    std::size_t destination_byte, std::size_t elements) {
                    const std::size_t plane_index =
                            callback_index++ / callbacks_per_plane;
                    const std::size_t host_bit =
                            (plane_index * elements_per_plane
                                    + row * columns + column)
                            * bits;
                    cpu_detail::copy_tile_row(
                            storage, destination_byte * 8, host,
                            host_bit, elements, bits, shift_table, mask_table);
                });
    }

    void region_to_host(
            const TensorView& source, std::span<std::byte> destination,
            RawWorkspaceView) const override {
        auto* host = reinterpret_cast<unsigned char*>(destination.data());
        const auto* storage = static_cast<const unsigned char*>(address_);
        const std::size_t bits = detail::leaf_bits(source.spec().data_type);
        if (bits % 8 != 0) {
            const std::size_t tail_bits =
                    (source.spec().shape.element_count() % 8)
                    * (bits % 8) % 8;
            if (tail_bits != 0) {
                std::fill_n(destination.data() + (destination.size() - 1),
                            1, std::byte{0});
            }
        }
        const std::size_t columns = source.spec().shape.dimensions().back();
        const auto dimensions = source.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t tile_columns =
                columns / TensorSpec::TILE + (columns % TensorSpec::TILE != 0);
        const std::size_t callbacks_per_plane = rows * tile_columns;
        const std::size_t elements_per_plane = rows * columns;
        std::size_t callback_index = 0;
        std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
        std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
        if (bits % 8 != 0) {
            for (std::size_t index = 0; index < TensorSpec::TILE; ++index) {
                shift_table[index] =
                        (index * bits) % (sizeof(std::uint32_t) * 8);
                mask_table[index] =
                        ((1u << bits) - 1u) << shift_table[index];
            }
        }
        cpu_detail::for_each_tile(
                source,
                [&](std::size_t row, std::size_t column,
                    std::size_t source_byte, std::size_t elements) {
                    const std::size_t plane_index =
                            callback_index++ / callbacks_per_plane;
                    const std::size_t host_bit =
                            (plane_index * elements_per_plane
                                    + row * columns + column)
                            * bits;
                    cpu_detail::copy_tile_row(
                            host, host_bit, storage, source_byte * 8,
                            elements, bits, shift_table, mask_table);
                });
    }

    CpuDevice& device_;
    Allocator& allocator_;
    void* address_ = nullptr;
};

}  // namespace

std::unique_ptr<Tensor> CpuDevice::create_tensor(const TensorSpec& spec) {
    return std::make_unique<CpuTensor>(spec, *this, allocator_);
}

}  // namespace iom
