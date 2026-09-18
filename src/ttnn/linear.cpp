#include "linear.hpp"
#include "copy.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/span.hpp>
#include <ttnn/tensor/tensor.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace iom::ttnn_detail {
namespace {

#ifndef IOM_TTNN_LINEAR_READER_KERNEL_SOURCE
#define IOM_TTNN_LINEAR_READER_KERNEL_SOURCE \
    "src/ttnn/kernels/linear_reader.cpp"
#endif
#ifndef IOM_TTNN_LINEAR_COMPUTE_KERNEL_SOURCE
#define IOM_TTNN_LINEAR_COMPUTE_KERNEL_SOURCE \
    "src/ttnn/kernels/linear_compute.cpp"
#endif
#ifndef IOM_TTNN_LINEAR_WRITER_KERNEL_SOURCE
#define IOM_TTNN_LINEAR_WRITER_KERNEL_SOURCE \
    "src/ttnn/kernels/linear_writer.cpp"
#endif

// Two tiles per operand buffer let the reader run one tile ahead of the matrix
// engine, and two scratch pages hold the at-most-two native source tiles a
// 32-row window can span. BF16 is the only leaf this port implements, so every
// page is exactly one 2048-byte native tile.
constexpr std::uint32_t kOperandTiles = 2;
constexpr std::uint32_t kScratchTiles = 2;

[[nodiscard]] std::uint32_t checked_u32(
        std::uint64_t value, const char* what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
                std::string("TTNN linear ") + what
                + " exceeds the native uint32_t range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t checked_dimension(
        std::size_t value, const char* what) {
    return checked_u32(static_cast<std::uint64_t>(value), what);
}

// Native geometry of one per-plane TTNN tensor: its DRAM base address, page
// size, padded tile grid, and carrier width. All of it is checked before any
// runtime argument is formed.
struct NativePlaneFacts {
    std::uint32_t address = 0;
    std::uint32_t page_size = 0;
    std::uint32_t padded_rows = 0;
    std::uint32_t padded_columns = 0;
};

[[nodiscard]] NativePlaneFacts plane_facts(
        const ttnn::Tensor& plane, const char* name) {
    auto* const buffer = plane.buffer();
    if (buffer == nullptr || !buffer->is_dram() || buffer->address() == 0) {
        throw std::runtime_error(
                std::string("TTNN linear ") + name
                + " has no native DRAM buffer");
    }
    const std::uint64_t expected_page =
            static_cast<std::uint64_t>(kLinearTileBytes);
    const std::uint32_t page = checked_u32(
            static_cast<std::uint64_t>(buffer->page_size()), "native page");
    if (page != expected_page) {
        throw std::runtime_error(
                std::string("TTNN linear ") + name
                + " native page is not one BF16 tiled carrier page");
    }
    const auto padded = plane.padded_shape();
    const std::uint32_t rows = checked_u32(
            static_cast<std::size_t>(padded[-2]), "padded rows");
    const std::uint32_t columns = checked_u32(
            static_cast<std::size_t>(padded[-1]), "padded columns");
    if (rows == 0 || columns == 0 || rows % 32 != 0 || columns % 32 != 0) {
        throw std::runtime_error(
                std::string("TTNN linear ") + name
                + " native tile geometry is malformed");
    }
    if (carrier_bytes(plane.dtype()) * kLinearTileCells != expected_page) {
        throw std::runtime_error(
                std::string("TTNN linear ") + name
                + " native carrier width is not BF16");
    }
    return NativePlaneFacts{buffer->address(), page, rows, columns};
}

}  // namespace

LinearProgram::LinearProgram(
        tt::tt_metal::distributed::MeshDevice& device)
        : workload_(), device_range_(device.shape()), core_(0, 0) {
    tt::tt_metal::Program native_program = tt::tt_metal::CreateProgram();
    const auto buffer_config =
            [](std::uint32_t cb, std::uint32_t tiles) {
                tt::tt_metal::CircularBufferConfig config(
                        tiles * kLinearTileBytes,
                        {{cb, tt::DataFormat::Float16_b}});
                config.set_page_size(cb, kLinearTileBytes);
                return config;
            };
    tt::tt_metal::CreateCircularBuffer(
            native_program, core_, buffer_config(kLinearIn0Cb, kOperandTiles));
    tt::tt_metal::CreateCircularBuffer(
            native_program, core_, buffer_config(kLinearIn1Cb, kOperandTiles));
    tt::tt_metal::CreateCircularBuffer(
            native_program, core_, buffer_config(kLinearOutCb, kOperandTiles));
    tt::tt_metal::CreateCircularBuffer(
            native_program, core_, buffer_config(kLinearScratchCb, kScratchTiles));

    std::vector<std::uint32_t> reader_compile_args{
            kLinearIn0Cb, kLinearIn1Cb, kLinearOutCb, kLinearScratchCb};
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(
            reader_compile_args);
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(
            reader_compile_args);
    reader_ = tt::tt_metal::CreateKernel(
            native_program, IOM_TTNN_LINEAR_READER_KERNEL_SOURCE, core_,
            tt::tt_metal::ReaderDataMovementConfig{reader_compile_args});

    std::vector<std::uint32_t> writer_compile_args{kLinearOutCb};
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(
            writer_compile_args);
    writer_ = tt::tt_metal::CreateKernel(
            native_program, IOM_TTNN_LINEAR_WRITER_KERNEL_SOURCE, core_,
            tt::tt_metal::WriterDataMovementConfig{writer_compile_args});

    compute_ = tt::tt_metal::CreateKernel(
            native_program, IOM_TTNN_LINEAR_COMPUTE_KERNEL_SOURCE, core_,
            tt::tt_metal::ComputeConfig{
                    .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .compile_args = {kLinearIn0Cb, kLinearIn1Cb, kLinearOutCb}});

    workload_.add_program(device_range_, std::move(native_program));
    auto& programs = workload_.get_programs();
    if (programs.size() != 1) {
        throw std::logic_error(
                "TTNN linear workload has an invalid program count");
    }
    program_ = &programs.begin()->second;
}

void LinearProgram::dispatch(
        tt::tt_metal::distributed::MeshCommandQueue& queue,
        const std::array<std::uint32_t, kLinearReaderRuntimeArgs>& reader_args,
        const std::array<std::uint32_t, kLinearComputeRuntimeArgs>& compute_args,
        const std::array<std::uint32_t, kLinearWriterRuntimeArgs>& writer_args) {
    tt::tt_metal::SetRuntimeArgs(
            *program_, reader_, core_,
            ttsl::Span<const std::uint32_t>(reader_args.data(), reader_args.size()));
    tt::tt_metal::SetRuntimeArgs(
            *program_, compute_, core_,
            ttsl::Span<const std::uint32_t>(
                    compute_args.data(), compute_args.size()));
    tt::tt_metal::SetRuntimeArgs(
            *program_, writer_, core_,
            ttsl::Span<const std::uint32_t>(writer_args.data(), writer_args.size()));
    tt::tt_metal::distributed::EnqueueMeshWorkload(
            queue, workload_, /*blocking=*/false);
}

void linear_planes(
        tt::tt_metal::distributed::MeshDevice& device, LinearProgram& program,
        const LinearNativeRequest& request, bool& any_submitted) {
    any_submitted = false;
    const std::span<const std::size_t> x_dimensions =
            request.x.spec.shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            request.w.spec.shape.dimensions();
    const std::size_t inner =
            x_dimensions[x_dimensions.size() - 1];
    const std::size_t output_features = w_dimensions[0];
    const bool head_planar =
            request.layout == LinearOutputLayout::head_planar;
    const std::size_t heads = head_planar ? request.heads : 1;
    const std::size_t head_dim =
            head_planar ? request.head_dim : output_features;
    const std::size_t planes = snapshot_plane_count(request.out.spec);
    if (heads == 0 || head_dim == 0 || planes == 0 || planes % heads != 0) {
        throw std::logic_error(
                "TTNN linear native plane geometry is inconsistent");
    }

    const auto* const x_planes =
            static_cast<const ttnn::Tensor*>(request.x.native_handle);
    const auto* const w_planes =
            static_cast<const ttnn::Tensor*>(request.w.native_handle);
    auto* const out_planes =
            static_cast<ttnn::Tensor*>(request.out.native_handle);
    if (x_planes == nullptr || w_planes == nullptr || out_planes == nullptr) {
        throw std::runtime_error("TTNN linear native plane handle is null");
    }

    const std::uint32_t row_tiles = checked_dimension(
            (request.rows + 31) / 32, "row tile count");
    const std::uint32_t column_tiles = checked_dimension(
            (head_dim + 31) / 32, "column tile count");
    const std::uint32_t inner_tiles = checked_dimension(
            (inner + 31) / 32, "inner tile count");
    const std::uint32_t window_rows = checked_dimension(
            request.rows, "projected row count");
    const std::uint32_t inner_extent =
            checked_dimension(inner, "inner extent");
    const std::uint32_t outer_extent =
            checked_dimension(head_dim, "outer extent");
    const std::uint32_t row_start =
            checked_dimension(request.start_row, "selected row start");

    // The weight is the single rank-two `[O, I]` plane shared unchanged by
    // every leading plane and head.
    const NativePlaneFacts weight =
            plane_facts(w_planes[request.w.plane_offset], "weight");
    if (weight.padded_columns / 32 < inner_tiles) {
        throw std::runtime_error(
                "TTNN linear weight native geometry is too narrow");
    }

    for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::size_t x_plane = snapshot_owner_plane_at(
                request.x.spec, request.x.plane_offset, request.x.plane_strides,
                plane / heads);
        const std::size_t out_plane = snapshot_owner_plane_at(
                request.out.spec, request.out.plane_offset,
                request.out.plane_strides, plane);
        const NativePlaneFacts x =
                plane_facts(x_planes[x_plane], "input");
        const NativePlaneFacts out =
                plane_facts(out_planes[out_plane], "output");
        if (x.padded_columns / 32 < inner_tiles
                || x.padded_rows / 32 < (row_start + window_rows + 31) / 32) {
            throw std::runtime_error(
                    "TTNN linear input native geometry is too small");
        }
        if (out.padded_rows / 32 < row_tiles
                || out.padded_columns / 32 < column_tiles) {
            throw std::runtime_error(
                    "TTNN linear output native geometry is too small");
        }
        const std::size_t head = plane % heads;
        const std::size_t weight_row = head * head_dim;
        if (weight.padded_rows < weight_row + head_dim) {
            throw std::runtime_error(
                    "TTNN linear weight native geometry is too short");
        }
        const std::uint32_t weight_row_start =
                checked_u32(weight_row, "weight row start");
        const std::array<std::uint32_t, kLinearReaderRuntimeArgs> reader_args{
                x.address,
                x.page_size,
                x.padded_columns / 32,
                row_start,
                weight.address,
                weight.page_size,
                weight.padded_columns / 32,
                weight_row_start,
                window_rows,
                inner_extent,
                outer_extent,
                row_tiles,
                column_tiles,
                inner_tiles};
        const std::array<std::uint32_t, kLinearComputeRuntimeArgs> compute_args{
                row_tiles, column_tiles, inner_tiles};
        const std::array<std::uint32_t, kLinearWriterRuntimeArgs> writer_args{
                out.address,
                out.page_size,
                out.padded_columns / 32,
                row_tiles,
                column_tiles};
        program.dispatch(
                device.mesh_command_queue(0), reader_args, compute_args,
                writer_args);
        any_submitted = true;
    }
}

}  // namespace iom::ttnn_detail
