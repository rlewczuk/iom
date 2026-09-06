#include "iom/cpu/device.hpp"

#include <cassert>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iom/iom.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/detail/aligned_storage.hpp"
#include "registry_state.hpp"

namespace iom {

    namespace {

        constexpr std::array kCpuSupportedDataTypes = {
                iom::DataType::BOOL,
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
                iom::DataType::I64, iom::DataType::U64,
                iom::DataType::F4_E2M1,
                iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
                iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
                iom::DataType::F8_E8M0,
                iom::DataType::F16, iom::DataType::BF16,
                iom::DataType::F32, iom::DataType::F64,
        };

        // Section-3 host encoding: each logical element is one field of
        // exactly leaf_bits(type) bits, laid out least-significant bit
        // first at bit offset (element index) * bits. Multi-byte fields
        // are byte-aligned little-endian, which is the same bit stream.
        std::uint64_t read_bits(
                const unsigned char* base, std::size_t bit_offset,
                std::size_t nbits) {
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < nbits; ++i) {
                const std::size_t bit = bit_offset + i;
                value |= static_cast<std::uint64_t>(
                                 (base[bit / 8] >> (bit % 8)) & 1)
                         << i;
            }
            return value;
        }

        void write_bits(
                unsigned char* base, std::size_t bit_offset,
                std::size_t nbits, std::uint64_t value) {
            for (std::size_t i = 0; i < nbits; ++i) {
                const std::size_t bit = bit_offset + i;
                unsigned char& byte = base[bit / 8];
                const unsigned char mask =
                        static_cast<unsigned char>(1u << (bit % 8));
                if ((value >> i) & 1) {
                    byte |= mask;
                } else {
                    byte &= static_cast<unsigned char>(~mask);
                }
            }
        }

        // Copies one encoded field between two bit-addressed byte buffers.
        // Every storage and host buffer carries the identical encoding, so
        // byte-aligned fields move as whole bytes and sub-byte fields move
        // bit by bit. No numeric conversion happens anywhere.
        void copy_value(
                unsigned char* destination, std::size_t destination_bit,
                const unsigned char* source, std::size_t source_bit,
                std::size_t nbits) {
            if (nbits % 8 == 0) {
                unsigned char* destination_bytes =
                        destination + destination_bit / 8;
                const unsigned char* source_bytes = source + source_bit / 8;
                if (destination_bytes != source_bytes) {
                    std::memcpy(destination_bytes, source_bytes, nbits / 8);
                }
                return;
            }
            write_bits(
                    destination, destination_bit, nbits,
                    read_bits(source, source_bit, nbits));
        }

        template <std::size_t kElementBytes>
        static inline void copy_tile_row_byte_aligned(
                unsigned char* destination, const unsigned char* source,
                std::size_t elements) {
            assert(elements <= TensorSpec::TILE);
            for (std::size_t index = 0; index < elements; ++index) {
                if (destination + index * kElementBytes
                        != source + index * kElementBytes) {
                    std::memcpy(
                            destination + index * kElementBytes,
                            source + index * kElementBytes,
                            kElementBytes);
                }
            }
        }

        static inline void copy_tile_row_subbyte(
                unsigned char* destination, std::size_t destination_bit,
                const unsigned char* source, std::size_t source_bit,
                std::size_t elements, std::size_t leaf_bits,
                std::array<std::uint32_t, TensorSpec::TILE>& shift_table,
                std::array<std::uint32_t, TensorSpec::TILE>& mask_table) {
            assert(elements <= TensorSpec::TILE);
            const std::size_t total_bits = elements * leaf_bits;
            const bool word_aligned =
                    source_bit % (sizeof(std::uint32_t) * 8) == 0
                    && destination_bit % (sizeof(std::uint32_t) * 8) == 0;
            if (word_aligned) {
                const std::size_t whole_word_bits =
                        total_bits / (sizeof(std::uint32_t) * 8)
                        * (sizeof(std::uint32_t) * 8);
                const std::size_t word_elements =
                        whole_word_bits % leaf_bits == 0
                        ? whole_word_bits / leaf_bits
                        : 0;
                const std::size_t word_count =
                        word_elements * leaf_bits
                        / (sizeof(std::uint32_t) * 8);
                std::uint32_t covered_mask = 0;
                for (std::size_t element_index = 0;
                     element_index < word_elements; ++element_index) {
                    covered_mask |= mask_table[element_index];
                    covered_mask |=
                            (1u << shift_table[element_index])
                            & mask_table[element_index];
                }
                for (std::size_t word_index = 0;
                     word_index < word_count; ++word_index) {
                    std::uint32_t source_word = 0;
                    std::memcpy(
                            &source_word,
                            source + source_bit / 8
                                    + word_index * sizeof(std::uint32_t),
                            sizeof(std::uint32_t));
                    const std::uint32_t destination_word =
                            source_word & covered_mask;
                    std::memcpy(
                            destination + destination_bit / 8
                                    + word_index * sizeof(std::uint32_t),
                            &destination_word, sizeof(std::uint32_t));
                }
                for (std::size_t index = word_elements;
                     index < elements; ++index) {
                    copy_value(
                            destination, destination_bit + index * leaf_bits,
                            source, source_bit + index * leaf_bits,
                            leaf_bits);
                }
                return;
            }

            for (std::size_t index = 0; index < elements; ++index) {
                copy_value(
                        destination, destination_bit + index * leaf_bits,
                        source, source_bit + index * leaf_bits,
                        leaf_bits);
            }
        }

        template <typename Op>
        static inline void for_each_tile(
                const TensorView& view, Op&& op) {
            const TensorSpec& spec = view.spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::size_t rows = dimensions[leading_rank];
            const std::size_t columns = dimensions[leading_rank + 1];
            const std::size_t tile_rows =
                    rows / TensorSpec::TILE
                    + (rows % TensorSpec::TILE != 0);
            const std::size_t tile_columns =
                    columns / TensorSpec::TILE
                    + (columns % TensorSpec::TILE != 0);
            const std::size_t bits = detail::leaf_bits(spec.data_type);
            const std::span<const std::size_t> strides =
                    view.plane_strides();

            auto visit = [&](
                                 auto&& self, std::size_t leading_index,
                                 std::size_t plane) -> void {
                if (leading_index != leading_rank) {
                    for (std::size_t index = 0;
                         index < dimensions[leading_index]; ++index) {
                        self(
                                self, leading_index + 1,
                                plane + index * strides[leading_index]);
                    }
                    return;
                }

                const std::size_t tile_bytes =
                        TensorSpec::TILE * TensorSpec::TILE * bits / 8;
                for (std::size_t tile_row = 0;
                     tile_row < tile_rows; ++tile_row) {
                    const std::size_t first_row =
                            tile_row * TensorSpec::TILE;
                    const std::size_t tile_row_byte =
                            detail::standard_plane_slot(
                                    spec, plane, first_row, 0)
                            * bits / 8;
                    for (std::size_t row_in_tile = 0;
                         row_in_tile < TensorSpec::TILE; ++row_in_tile) {
                        const std::size_t row = first_row + row_in_tile;
                        if (row >= rows) {
                            break;
                        }
                        const std::size_t row_byte =
                                tile_row_byte + row_in_tile * TensorSpec::TILE
                                        * bits / 8;
                        for (std::size_t tile_column = 0;
                             tile_column < tile_columns; ++tile_column) {
                            const std::size_t column =
                                    tile_column * TensorSpec::TILE;
                            const std::size_t remaining = columns - column;
                            const std::size_t elements =
                                    remaining < TensorSpec::TILE
                                    ? remaining
                                    : TensorSpec::TILE;
                            op(
                                    row, column,
                                    row_byte + tile_column * tile_bytes,
                                    elements);
                        }
                    }
                }
            };
            visit(visit, 0, view.plane_offset());
        }

        template <typename Op>
        static inline void for_each_tile_lockstep(
                const TensorView& source, const TensorView& destination,
                Op&& op) {
            const TensorSpec& source_spec = source.spec();
            const TensorSpec& destination_spec = destination.spec();
            const std::span<const std::size_t> source_dimensions =
                    source_spec.shape.dimensions();
            const std::size_t leading_rank = source_dimensions.size() - 2;
            const std::size_t rows = source_dimensions[leading_rank];
            const std::size_t columns =
                    source_dimensions[leading_rank + 1];
            const std::size_t tile_rows =
                    rows / TensorSpec::TILE
                    + (rows % TensorSpec::TILE != 0);
            const std::size_t tile_columns =
                    columns / TensorSpec::TILE
                    + (columns % TensorSpec::TILE != 0);
            const std::size_t source_bits =
                    detail::leaf_bits(source_spec.data_type);
            const std::size_t destination_bits =
                    detail::leaf_bits(destination_spec.data_type);
            const std::span<const std::size_t> source_strides =
                    source.plane_strides();
            const std::span<const std::size_t> destination_strides =
                    destination.plane_strides();
            bool identical_layout =
                    source.plane_offset() == destination.plane_offset()
                    && source_strides.size() == destination_strides.size();
            for (std::size_t index = 0;
                 identical_layout && index < source_strides.size(); ++index) {
                identical_layout =
                        source_strides[index] == destination_strides[index];
            }

            auto visit = [&](
                                 auto&& self, std::size_t leading_index,
                                 std::size_t source_plane,
                                 std::size_t destination_plane) -> void {
                if (leading_index != leading_rank) {
                    for (std::size_t index = 0;
                         index < source_dimensions[leading_index]; ++index) {
                        self(
                                self, leading_index + 1,
                                source_plane
                                        + index
                                                * source_strides[leading_index],
                                destination_plane
                                        + index
                                                * destination_strides[
                                                        leading_index]);
                    }
                    return;
                }

                const std::size_t source_tile_bytes =
                        TensorSpec::TILE * TensorSpec::TILE * source_bits / 8;
                const std::size_t destination_tile_bytes =
                        TensorSpec::TILE * TensorSpec::TILE
                        * destination_bits / 8;
                for (std::size_t tile_row = 0;
                     tile_row < tile_rows; ++tile_row) {
                    const std::size_t first_row =
                            tile_row * TensorSpec::TILE;
                    const std::size_t source_tile_row_byte =
                            detail::standard_plane_slot(
                                    source_spec, source_plane, first_row, 0)
                            * source_bits / 8;
                    const std::size_t destination_tile_row_byte =
                            identical_layout
                            ? source_tile_row_byte
                            : detail::standard_plane_slot(
                                      destination_spec, destination_plane,
                                      first_row, 0)
                                      * destination_bits / 8;
                    for (std::size_t row_in_tile = 0;
                         row_in_tile < TensorSpec::TILE; ++row_in_tile) {
                        const std::size_t row = first_row + row_in_tile;
                        if (row >= rows) {
                            break;
                        }
                        const std::size_t source_row_byte =
                                source_tile_row_byte
                                + row_in_tile * TensorSpec::TILE
                                        * source_bits / 8;
                        const std::size_t destination_row_byte =
                                destination_tile_row_byte
                                + row_in_tile * TensorSpec::TILE
                                        * destination_bits / 8;
                        for (std::size_t tile_column = 0;
                             tile_column < tile_columns; ++tile_column) {
                            const std::size_t column =
                                    tile_column * TensorSpec::TILE;
                            const std::size_t remaining = columns - column;
                            const std::size_t elements =
                                    remaining < TensorSpec::TILE
                                    ? remaining
                                    : TensorSpec::TILE;
                            op(
                                    row, column,
                                    source_row_byte
                                            + tile_column * source_tile_bytes,
                                    destination_row_byte
                                            + tile_column
                                                    * destination_tile_bytes,
                                    elements);
                        }
                    }
                }
            };
            visit(
                    visit, 0, source.plane_offset(),
                    destination.plane_offset());
        }

        // Calls op(owner_plane, row, column, linear_index) for every
        // logical element of the view in row-major coordinate order. The
        // view's plane offset and strides carry the leading coordinates.
        template <typename Op>
        void for_each_coordinate(const TensorView& view, Op&& op) {
            const TensorSpec& spec = view.spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::size_t rows = dimensions[leading_rank];
            const std::size_t columns = dimensions[leading_rank + 1];
            const std::size_t count = spec.shape.element_count();
            const std::span<const std::size_t> strides =
                    view.plane_strides();

            for (std::size_t linear = 0; linear < count; ++linear) {
                std::size_t rest = linear;
                const std::size_t column = rest % columns;
                rest /= columns;
                const std::size_t row = rest % rows;
                rest /= rows;

                // View transforms keep every addressed owner plane in
                // bounds, so this accumulation cannot overflow.
                std::size_t plane = view.plane_offset();
                for (std::size_t k = leading_rank; k-- > 0;) {
                    plane += (rest % dimensions[k]) * strides[k];
                    rest /= dimensions[k];
                }
                op(plane, row, column, linear);
            }
        }

        // Owner plane of the view's linear_index-th logical element; the
        // row-major inverse of for_each_coordinate's plane accumulation.
        std::size_t plane_at(const TensorView& view, std::size_t linear) {
            const TensorSpec& spec = view.spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::size_t rows = dimensions[leading_rank];
            const std::size_t columns = dimensions[leading_rank + 1];
            const std::span<const std::size_t> strides =
                    view.plane_strides();

            std::size_t rest = linear / (rows * columns);
            std::size_t plane = view.plane_offset();
            for (std::size_t k = leading_rank; k-- > 0;) {
                plane += (rest % dimensions[k]) * strides[k];
                rest /= dimensions[k];
            }
            return plane;
        }

    }  // namespace

    class CpuDevice final : public Device {
    public:
        explicit CpuDevice(Allocator& allocator)
                : allocator_(allocator) {}

        ~CpuDevice() override {
            registry_state_.quarantine.drain();
        }

        [[nodiscard]] BackendKind backend_kind() const noexcept override {
            return BackendKind::CPU;
        }

        [[nodiscard]] std::uint32_t backend_device() const noexcept override {
            return 0;
        }
        [[nodiscard]] std::span<const iom::DataType>
                supported_data_types() const noexcept override {
            return {kCpuSupportedDataTypes.data(),
                    kCpuSupportedDataTypes.size()};
        }

        [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                const TensorSpec& spec) override;
        [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

        [[nodiscard]] cpu_detail::CpuRegistryState& registry_state() noexcept {
            return registry_state_;
        }

    private:
        cpu_detail::CpuRegistryState registry_state_;
        Allocator& allocator_;
    };

    /**
     * Owner of one standard-layout allocation. The Tensor base validates
     * the specification before the body allocates; every construction
     * failure after allocation frees the storage before propagating.
     */
    class CpuTensor final : public Tensor {
    public:
        CpuTensor(const TensorSpec& spec, CpuDevice& device,
                  Allocator& allocator)
                : Tensor(spec, device), device_(device),
                  allocator_(allocator) {
            address_ = iom::detail::allocate_aligned_storage(
                    allocator_,
                    view().spec().tiled_storage_nbytes(),
                    [] {},
                    "CPU tensor storage is not 32-byte aligned");
        }

        ~CpuTensor() noexcept override {
            if (address_ == nullptr) {
                return;
            }

            const std::size_t bytes = view().spec().tiled_storage_nbytes();
            const auto quarantine_storage = [this, bytes]() noexcept {
                try {
                    device_.registry_state().quarantine
                            .emplace<detail::AllocatorCleanupAction>(
                                    allocator_, address_, bytes);
                } catch (...) {
                    // Leaking is safer than returning failed storage to the
                    // allocator when quarantine allocation itself fails.
                }
                address_ = nullptr;
            };
            const auto release_storage = [this]() noexcept {
                iom::detail::release_aligned_storage(allocator_, address_);
            };
            iom::detail::release_or_quarantine(
                    device_.registry_state().registry, address_,
                    quarantine_storage, release_storage);
        }

    private:
        [[nodiscard]] void* storage_handle() noexcept override {
            return address_;
        }

        void region_from_host(
                const TensorView& destination,
                std::span<const std::byte> source) override {
            unsigned char* storage =
                    static_cast<unsigned char*>(address_);
            const auto* host = reinterpret_cast<const unsigned char*>(
                    source.data());
            const std::size_t bits =
                    detail::leaf_bits(destination.spec().data_type);
            const std::size_t columns =
                    destination.spec().shape.dimensions().back();
            const std::span<const std::size_t> dimensions =
                    destination.spec().shape.dimensions();
            const std::size_t rows = dimensions[dimensions.size() - 2];
            const std::size_t tile_columns =
                    columns / TensorSpec::TILE
                    + (columns % TensorSpec::TILE != 0);
            const std::size_t row_runs_per_plane = tile_columns;
            const std::size_t callbacks_per_plane =
                    rows * row_runs_per_plane;
            const std::size_t elements_per_plane = rows * columns;
            std::size_t callback_index = 0;
            std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
            std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
            if (bits % 8 != 0) {
                for (std::size_t index = 0;
                     index < TensorSpec::TILE; ++index) {
                    shift_table[index] =
                            (index * bits) % (sizeof(std::uint32_t) * 8);
                    mask_table[index] =
                            ((1u << bits) - 1u) << shift_table[index];
                }
            }
            for_each_tile(
                    destination,
                    [&](std::size_t row, std::size_t column,
                        std::size_t destination_byte,
                        std::size_t elements) {
                        const std::size_t plane_index =
                                callback_index / callbacks_per_plane;
                        ++callback_index;
                        const std::size_t host_bit =
                                (plane_index * elements_per_plane
                                        + row * columns + column)
                                * bits;
                        if (bits % 8 == 0) {
                            const auto* row_source =
                                    host + host_bit / 8;
                            auto* row_destination =
                                    storage + destination_byte;
                            switch (bits) {
                                case 8:
                                    copy_tile_row_byte_aligned<1>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 16:
                                    copy_tile_row_byte_aligned<2>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 32:
                                    copy_tile_row_byte_aligned<4>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 64:
                                    copy_tile_row_byte_aligned<8>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                default:
                                    throw std::invalid_argument(
                                            "unsupported byte-aligned CPU leaf width");
                            }
                        } else {
                            copy_tile_row_subbyte(
                                    storage, destination_byte * 8,
                                    host, host_bit, elements, bits,
                                    shift_table, mask_table);
                        }
                    });
        }

        void region_to_host(
                const TensorView& source,
                std::span<std::byte> destination) const override {
            unsigned char* host =
                    reinterpret_cast<unsigned char*>(destination.data());
            const unsigned char* storage =
                    static_cast<const unsigned char*>(address_);
            const std::size_t bits =
                    detail::leaf_bits(source.spec().data_type);
            if (bits % 8 != 0) {
                const std::size_t tail_bits =
                        (source.spec().shape.element_count() % 8)
                        * (bits % 8) % 8;
                if (tail_bits != 0) {
                    std::fill_n(
                            destination.data() + (destination.size() - 1),
                            1, std::byte{0});
                }
            }
            const std::size_t columns =
                    source.spec().shape.dimensions().back();
            const std::span<const std::size_t> dimensions =
                    source.spec().shape.dimensions();
            const std::size_t rows = dimensions[dimensions.size() - 2];
            const std::size_t tile_columns =
                    columns / TensorSpec::TILE
                    + (columns % TensorSpec::TILE != 0);
            const std::size_t row_runs_per_plane = tile_columns;
            const std::size_t callbacks_per_plane =
                    rows * row_runs_per_plane;
            const std::size_t elements_per_plane = rows * columns;
            std::size_t callback_index = 0;
            std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
            std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
            if (bits % 8 != 0) {
                for (std::size_t index = 0;
                     index < TensorSpec::TILE; ++index) {
                    shift_table[index] =
                            (index * bits) % (sizeof(std::uint32_t) * 8);
                    mask_table[index] =
                            ((1u << bits) - 1u) << shift_table[index];
                }
            }
            for_each_tile(
                    source,
                    [&](std::size_t row, std::size_t column,
                        std::size_t source_byte, std::size_t elements) {
                        const std::size_t plane_index =
                                callback_index / callbacks_per_plane;
                        ++callback_index;
                        const std::size_t host_bit =
                                (plane_index * elements_per_plane
                                        + row * columns + column)
                                * bits;
                        auto* row_destination = host + host_bit / 8;
                        const auto* row_source = storage + source_byte;
                        if (bits % 8 == 0) {
                            switch (bits) {
                                case 8:
                                    copy_tile_row_byte_aligned<1>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 16:
                                    copy_tile_row_byte_aligned<2>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 32:
                                    copy_tile_row_byte_aligned<4>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                case 64:
                                    copy_tile_row_byte_aligned<8>(
                                            row_destination, row_source,
                                            elements);
                                    break;
                                default:
                                    throw std::invalid_argument(
                                            "unsupported byte-aligned CPU leaf width");
                            }
                        } else {
                            copy_tile_row_subbyte(
                                    host, host_bit, storage,
                                    source_byte * 8, elements, bits,
                                    shift_table, mask_table);
                        }
                    });
        }
        CpuDevice& device_;
        Allocator& allocator_;
        void* address_ = nullptr;
    };

    class CpuQueue final : public DeviceOps {
        struct Task {
            std::uint64_t sequence;
            TensorView source;
            TensorView destination;
            bool no_op;
            void* fence = nullptr;

            Task(std::uint64_t sequence_,
                 const TensorView& source_,
                 TensorView& destination_,
                 bool no_op_)
                : sequence(sequence_),
                  source(source_),
                  destination(destination_),
                  no_op(no_op_) {}
        };


    public:
        explicit CpuQueue(CpuDevice& device)
                : device_(&device),
                  worker_(
                          detail::StagedWorker<Task>::Callbacks{
                                  [this](Task& task) {
                                      execute(task);
                                  },
                                  [](void*) {},
                                  [](void*) {},
                                  [this](
                                          std::uint64_t sequence,
                                          std::exception_ptr failure) {
                                      complete(
                                              sequence, std::move(failure));
                                  }},
                          detail::StagedWorker<Task>::PublishPolicy::
                                  CompleteOnThrow) {
            worker_.start();
        }

        ~CpuQueue() override {
            worker_.shutdown_and_drain();
        }



        oid copy(const TensorView& source, TensorView& destination) override {
            std::lock_guard<std::mutex> submission_lock(
                    submission_order_mutex_);
            validate_copy(*device_, source, destination);
            const bool no_op = identical_window(source, destination);
            return submit(
                    [this, &source, &destination, no_op](
                            std::uint64_t sequence) {
                        Task task(sequence, source, destination, no_op);
                        worker_.submit_copy(std::move(task));
                    });
        }

        [[nodiscard]] std::string_view backend_label() const noexcept override {
            return "CPU";
        }

    private:
        void execute(Task& task) {
            if (!task.no_op) {
                copy_elements(task.source, task.destination);
            }
        }

        static void copy_elements(
                const TensorView& source, TensorView& destination) {
            const auto* source_base = static_cast<const unsigned char*>(
                    source.native_handle());
            auto* destination_base = static_cast<unsigned char*>(
                    destination.native_handle());
            const std::size_t bits =
                    detail::leaf_bits(source.spec().data_type);
            std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
            std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
            if (bits % 8 != 0) {
                for (std::size_t index = 0;
                     index < TensorSpec::TILE; ++index) {
                    shift_table[index] =
                            (index * bits) % (sizeof(std::uint32_t) * 8);
                    mask_table[index] =
                            ((1u << bits) - 1u) << shift_table[index];
                }
            }
            for_each_tile_lockstep(
                    source, destination,
                    [&](std::size_t, std::size_t,
                        std::size_t source_byte,
                        std::size_t destination_byte,
                        std::size_t elements) {
                        if (bits % 8 == 0) {
                            switch (bits) {
                                case 8:
                                    copy_tile_row_byte_aligned<1>(
                                            destination_base + destination_byte,
                                            source_base + source_byte,
                                            elements);
                                    break;
                                case 16:
                                    copy_tile_row_byte_aligned<2>(
                                            destination_base + destination_byte,
                                            source_base + source_byte,
                                            elements);
                                    break;
                                case 32:
                                    copy_tile_row_byte_aligned<4>(
                                            destination_base + destination_byte,
                                            source_base + source_byte,
                                            elements);
                                    break;
                                case 64:
                                    copy_tile_row_byte_aligned<8>(
                                            destination_base + destination_byte,
                                            source_base + source_byte,
                                            elements);
                                    break;
                                default:
                                    throw std::invalid_argument(
                                            "unsupported byte-aligned CPU leaf width");
                            }
                        } else {
                            copy_tile_row_subbyte(
                                    destination_base, destination_byte * 8,
                                    source_base, source_byte * 8,
                                    elements, bits, shift_table, mask_table);
                        }
                    });
        }

        CpuDevice* device_;
        std::mutex submission_order_mutex_;
        detail::StagedWorker<Task> worker_;
    };

    std::unique_ptr<Tensor> CpuDevice::create_tensor(const TensorSpec& spec) {
        return std::make_unique<CpuTensor>(spec, *this, allocator_);
    }

    std::unique_ptr<DeviceOps> CpuDevice::create_ops() {
        return std::make_unique<CpuQueue>(*this);
    }

    std::unique_ptr<Device> make_cpu_device(Allocator& allocator) {
        return std::make_unique<CpuDevice>(allocator);
    }

}  // namespace iom
