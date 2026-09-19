#include "embedding.hpp"
#include "copy.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>

namespace iom::ttnn_detail {
namespace {

constexpr std::size_t kNativeTileCells = 32 * 32;
constexpr std::size_t kNativeTileBytes = 4096;
constexpr std::uint32_t kEmbeddingCb = 0;

#ifndef IOM_TTNN_EMBEDDING_KERNEL_SOURCE
#define IOM_TTNN_EMBEDDING_KERNEL_SOURCE "src/ttnn/kernels/embedding.cpp"
#endif

[[nodiscard]] std::size_t plane_count(const EmbeddingSnapshot& view) {
    const std::span<const std::size_t> dimensions = view.spec.shape.dimensions();
    std::size_t count = 1;
    for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
        if (dimensions[i] != 0
                && count > std::numeric_limits<std::size_t>::max()
                                     / dimensions[i]) {
            throw std::overflow_error("TTNN embedding plane count overflows");
        }
        count *= dimensions[i];
    }
    return count;
}

[[nodiscard]] std::size_t owner_plane_at(
        const EmbeddingSnapshot& view, std::size_t index) {
    const std::span<const std::size_t> dimensions = view.spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    std::size_t plane = view.plane_offset;
    for (std::size_t k = leading_rank; k-- > 0;) {
        plane += (index % dimensions[k]) * view.plane_strides[k];
        index /= dimensions[k];
    }
    return plane;
}

[[nodiscard]] std::uint32_t checked_u32(
        std::uint64_t value, const char* what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
                std::string("TTNN embedding ") + what
                + " exceeds uint32_t runtime range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t checked_dimension(
        std::size_t value, const char* what) {
    return checked_u32(static_cast<std::uint64_t>(value), what);
}

[[nodiscard]] bool signed_id(DataType type) noexcept {
    switch (type) {
        case DataType::I2:
        case DataType::I4:
        case DataType::I8:
        case DataType::I16:
        case DataType::I32:
        case DataType::I64:
            return true;
        case DataType::U2:
        case DataType::U4:
        case DataType::U8:
        case DataType::U16:
        case DataType::U32:
        case DataType::U64:
        case DataType::BOOL:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F8_E8M0:
        case DataType::F16:
        case DataType::BF16:
        case DataType::F32:
        case DataType::F64:
            return false;
    }
    return false;
}

struct NativePlaneFacts {
    std::uint32_t address = 0;
    std::uint32_t page_size = 0;
    std::uint32_t padded_rows = 0;
    std::uint32_t padded_columns = 0;
    std::uint32_t carrier_bytes = 0;
};

[[nodiscard]] NativePlaneFacts plane_facts(
        const ttnn::Tensor& plane, const char* name) {
    auto* const buffer = plane.buffer();
    if (buffer == nullptr || !buffer->is_dram() || buffer->address() == 0) {
        throw std::runtime_error(
                std::string("TTNN embedding ") + name
                + " has no native DRAM buffer");
    }
    const std::uint32_t carrier = checked_dimension(
            carrier_bytes(plane.dtype()), "carrier width");
    const std::uint64_t expected_page =
            static_cast<std::uint64_t>(carrier) * kNativeTileCells;
    const std::uint32_t page = checked_u32(
            static_cast<std::uint64_t>(buffer->page_size()), "native page");
    if (page != expected_page || (page != 1024 && page != 2048 && page != 4096)) {
        throw std::runtime_error(
                std::string("TTNN embedding ") + name
                + " native page is not one tiled carrier page");
    }
    const auto padded = plane.padded_shape();
    const std::uint32_t rows = checked_dimension(
            static_cast<std::size_t>(padded[-2]), "padded rows");
    const std::uint32_t columns = checked_dimension(
            static_cast<std::size_t>(padded[-1]), "padded columns");
    if (rows == 0 || columns == 0 || rows % 32 != 0 || columns % 32 != 0) {
        throw std::runtime_error(
                std::string("TTNN embedding ") + name
                + " native tile geometry is malformed");
    }
    return NativePlaneFacts{
            buffer->address(), page, rows, columns, carrier};
}

}  // namespace

EmbeddingStatusSlot::EmbeddingStatusSlot()
        : transfers{tt::tt_metal::distributed::ShardDataTransfer{
                tt::tt_metal::distributed::MeshCoordinate{0, 0}}} {}

EmbeddingStatusResources::EmbeddingStatusResources(std::size_t capacity) {
    slots_.resize(std::max<std::size_t>(1, capacity));
}

std::size_t EmbeddingStatusResources::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t index = 0; index < slots_.size(); ++index) {
        EmbeddingStatusSlot& slot = slots_[index];
        if (!slot.in_use && !slot.retired) {
            slot.in_use = true;
            slot.packet.bytes.fill(std::byte{0});
            slot.transfers[0].host_data(slot.packet.bytes.data());
            return index;
        }
    }
    throw detail::AdmissionResourceUnavailable{};
}

void EmbeddingStatusResources::release(
        std::size_t slot, bool completion_proven) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slots_.size() || !slots_[slot].in_use) {
        return;
    }
    slots_[slot].in_use = false;
    if (!completion_proven) {
        slots_[slot].retired = true;
    }
}

EmbeddingStatusSlot& EmbeddingStatusResources::get(std::size_t slot) noexcept {
    return slots_[slot];
}

const EmbeddingStatusSlot& EmbeddingStatusResources::get(
        std::size_t slot) const noexcept {
    return slots_[slot];
}

EmbeddingProgram::EmbeddingProgram(
        tt::tt_metal::distributed::MeshDevice& device)
        : workload_(), device_range_(device.shape()), core_(0, 0) {
    tt::tt_metal::Program native_program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CircularBufferConfig cb_config(
            kNativeTileBytes,
            {{kEmbeddingCb, tt::DataFormat::UInt32}});
    cb_config.set_page_size(kEmbeddingCb, kNativeTileBytes);
    tt::tt_metal::CreateCircularBuffer(native_program, core_, cb_config);

    std::vector<std::uint32_t> compile_args{kEmbeddingCb};
    tt::tt_metal::TensorAccessorArgs accessor_args =
            tt::tt_metal::TensorAccessorArgs::create_dram_interleaved();
    accessor_args.append_to(compile_args);
    kernel_ = tt::tt_metal::CreateKernel(
            native_program, IOM_TTNN_EMBEDDING_KERNEL_SOURCE, core_,
            tt::tt_metal::ReaderDataMovementConfig{compile_args});
    workload_.add_program(device_range_, std::move(native_program));
    auto& programs = workload_.get_programs();
    if (programs.size() != 1) {
        throw std::logic_error("TTNN embedding workload has an invalid program count");
    }
    program_ = &programs.begin()->second;
}

void EmbeddingProgram::dispatch(
        tt::tt_metal::distributed::MeshCommandQueue& queue,
        const std::array<std::uint32_t, kEmbeddingRuntimeArgCount>& args) {
    tt::tt_metal::SetRuntimeArgs(
            *program_, kernel_, core_,
            ttsl::Span<const std::uint32_t>(args.data(), args.size()));
    tt::tt_metal::distributed::EnqueueMeshWorkload(
            queue, workload_, /*blocking=*/false);
}

void embedding_planes(
        tt::tt_metal::distributed::MeshDevice& device,
        EmbeddingProgram& program,
        EmbeddingStatusSlot& status,
        const EmbeddingNativeRequest& request,
        const NativeWorkspace& workspace,
        bool& any_submitted) {
    any_submitted = false;
    if (workspace.owner_handle == nullptr || workspace.page_size == 0) {
        throw std::runtime_error(
                "TTNN embedding status workspace has no native owner");
    }
    const std::uint32_t vocabulary = checked_dimension(
            request.table.spec.shape.dimension(0), "vocabulary extent");
    const std::uint32_t run = checked_dimension(
            request.indices.spec.shape.dimension(
                    request.indices.spec.shape.rank() - 1),
            "index run extent");
    const std::uint32_t features = checked_dimension(
            request.table.spec.shape.dimension(1), "feature extent");
    const std::uint32_t payload_bits = checked_dimension(
            detail::leaf_bits(request.out.spec.data_type), "payload width");
    const std::uint32_t id_bits = checked_dimension(
            detail::leaf_bits(request.indices.spec.data_type), "index width");
    // Native carrier columns per logical element. A 64-bit payload or index
    // leaf occupies two consecutive native columns (low word first), which is
    // exactly the expansion `TtnnTensor` applies to its native plane.
    const std::uint32_t table_factor = checked_dimension(
            carrier_factor(request.table.spec.data_type),
            "table carrier factor");
    const std::uint32_t index_factor = checked_dimension(
            carrier_factor(request.indices.spec.data_type),
            "index carrier factor");
    const std::uint32_t output_factor = checked_dimension(
            carrier_factor(request.out.spec.data_type),
            "output carrier factor");

    const auto* const table_planes =
            static_cast<const ttnn::Tensor*>(request.table.native_handle);
    const auto* const index_planes =
            static_cast<const ttnn::Tensor*>(request.indices.native_handle);
    auto* const output_planes =
            static_cast<ttnn::Tensor*>(request.out.native_handle);
    if (table_planes == nullptr || index_planes == nullptr
            || output_planes == nullptr) {
        throw std::runtime_error(
                "TTNN embedding native plane handle is null");
    }

    const NativePlaneFacts table =
            plane_facts(table_planes[request.table.plane_offset], "table");
    if (table.padded_rows < vocabulary
            || table.padded_columns
                    < static_cast<std::uint64_t>(features) * table_factor) {
        throw std::runtime_error(
                "TTNN embedding table native geometry is too small");
    }
    const std::size_t planes = plane_count(request.indices);
    if (planes != plane_count(request.out)) {
        throw std::invalid_argument(
                "TTNN embedding index and output plane counts differ");
    }

    status.transfers[0]
            .host_data(status.packet.bytes.data())
            .region(tt::tt_metal::BufferRegion{
                    checked_u32(workspace.offset, "status subrange"), 32});
    device.mesh_command_queue(0).enqueue_write_shards(
            workspace.owner_handle, status.transfers, /*blocking=*/false);
    any_submitted = true;

    const std::uint32_t id_signed_word = signed_id(
            request.indices.spec.data_type) ? 1u : 0u;
    const std::uint32_t status_base = checked_u32(
            workspace.base, "status base address");
    const std::uint32_t status_page = checked_u32(
            workspace.page_size, "status page size");
    for (std::size_t plane_index = 0; plane_index < planes; ++plane_index) {
        const std::size_t index_plane = owner_plane_at(
                request.indices, plane_index);
        const std::size_t output_plane = owner_plane_at(
                request.out, plane_index);
        const NativePlaneFacts indices =
                plane_facts(index_planes[index_plane], "indices");
        const NativePlaneFacts output =
                plane_facts(output_planes[output_plane], "output");
        if (indices.padded_rows < 1
                || indices.padded_columns
                        < static_cast<std::uint64_t>(run) * index_factor
                || output.padded_rows < run
                || output.padded_columns
                        < static_cast<std::uint64_t>(features)
                                  * output_factor) {
            throw std::runtime_error(
                    "TTNN embedding native geometry is too small");
        }
        std::array<std::uint32_t, kEmbeddingRuntimeArgCount> args{
                vocabulary,
                run,
                features,
                table.address,
                table.page_size,
                indices.address,
                indices.page_size,
                output.address,
                output.page_size,
                table.carrier_bytes,
                indices.carrier_bytes,
                output.carrier_bytes,
                payload_bits,
                id_bits,
                id_signed_word,
                table.padded_columns,
                indices.padded_columns,
                output.padded_columns,
                status_base,
                status_page,
                checked_u32(workspace.offset, "status subrange"),
                table_factor,
                index_factor};
        program.dispatch(device.mesh_command_queue(0), args);
        any_submitted = true;
    }
    device.mesh_command_queue(0).enqueue_read_shards(
            status.transfers, workspace.owner_handle, /*blocking=*/false);
    any_submitted = true;
}

}  // namespace iom::ttnn_detail
