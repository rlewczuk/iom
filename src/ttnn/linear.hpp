#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "device_internal.hpp"
#include "iom/iom.hpp"

namespace iom::ttnn_detail {

// One BF16 native tile: 32x32 carrier cells of two bytes each.
constexpr std::size_t kLinearTileCells = 32 * 32;
constexpr std::size_t kLinearTileBytes = kLinearTileCells * 2;
constexpr std::uint32_t kLinearIn0Cb = 0;
constexpr std::uint32_t kLinearIn1Cb = 1;
constexpr std::uint32_t kLinearScratchCb = 2;
constexpr std::uint32_t kLinearOutCb = 16;
constexpr std::size_t kLinearReaderRuntimeArgs = 14;
constexpr std::size_t kLinearComputeRuntimeArgs = 3;
constexpr std::size_t kLinearWriterRuntimeArgs = 5;

// Immutable admission snapshot of one native linear operand: value-copied
// view metadata and the stable owner and native-handle identities of that
// view. No native program retains the caller's borrowed view object.
struct LinearSnapshot {
    TensorSpec spec;
    void* native_handle = nullptr;
    std::size_t plane_offset = 0;
    std::vector<std::size_t> plane_strides;
};

// Immutable native linear request: the three operand snapshots plus the
// validated selected row window, output mode, and head scalars.
struct LinearNativeRequest {
    LinearSnapshot x;
    LinearSnapshot w;
    LinearSnapshot out;
    std::size_t start_row = 0;
    std::size_t rows = 0;
    LinearOutputLayout layout = LinearOutputLayout::ordinary;
    std::size_t heads = 0;
    std::size_t head_dim = 0;
};

// One reusable direct-Metalium program family per queue: a row-window and
// HF-orientation reader, the FP32-accumulating matrix engine, and a direct
// output writer, all on one Tensix core of the existing unit mesh. The program
// holds only release-visible state: the workload and its kernel handles are
// created once and every dispatch installs the plane's own runtime arguments.
class LinearProgram final {
public:
    explicit LinearProgram(tt::tt_metal::distributed::MeshDevice& device);
    LinearProgram(const LinearProgram&) = delete;
    LinearProgram& operator=(const LinearProgram&) = delete;

    void dispatch(
            tt::tt_metal::distributed::MeshCommandQueue& queue,
            const std::array<std::uint32_t, kLinearReaderRuntimeArgs>&
                    reader_args,
            const std::array<std::uint32_t, kLinearComputeRuntimeArgs>&
                    compute_args,
            const std::array<std::uint32_t, kLinearWriterRuntimeArgs>&
                    writer_args);

private:
    tt::tt_metal::distributed::MeshWorkload workload_;
    tt::tt_metal::distributed::MeshCoordinateRange device_range_;
    tt::tt_metal::CoreCoord core_;
    tt::tt_metal::KernelHandle reader_ = 0;
    tt::tt_metal::KernelHandle compute_ = 0;
    tt::tt_metal::KernelHandle writer_ = 0;
    tt::tt_metal::Program* program_ = nullptr;
};

// Enqueues one direct native program for every output plane of the request:
// ordinary planes are `[R, O]` and head-planar planes are the `[R, D]` planes
// of one head each, so the operation's own output planes are written in place.
void linear_planes(
        tt::tt_metal::distributed::MeshDevice& device, LinearProgram& program,
        const LinearNativeRequest& request, bool& any_submitted);

}  // namespace iom::ttnn_detail
