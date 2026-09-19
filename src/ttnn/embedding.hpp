#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "device_internal.hpp"
#include "iom/iom.hpp"

namespace iom::ttnn_detail {

struct EmbeddingSnapshot {
    TensorSpec spec;
    void* native_handle = nullptr;
    std::size_t plane_offset = 0;
    std::vector<std::size_t> plane_strides;
};

struct EmbeddingNativeRequest {
    EmbeddingSnapshot table;
    EmbeddingSnapshot indices;
    EmbeddingSnapshot out;
    NativeWorkspace native_workspace;
    detail::WorkspaceLease workspace_lease;
};

// Runtime words: V, R, F, native table/index/output addresses and pages, the
// three native carrier widths, the payload and index widths, the signed-index
// marker, the three padded native column counts, the status owner's base, page
// size, and subrange, and the table and index native carrier factors (one
// column per logical element, or two consecutive columns for a 64-bit leaf).
constexpr std::size_t kEmbeddingRuntimeArgCount = 23;

struct alignas(32) EmbeddingStatusPacket {
    std::array<std::byte, 32> bytes{};
};

struct EmbeddingStatusSlot {
    EmbeddingStatusPacket packet;
    std::vector<tt::tt_metal::distributed::ShardDataTransfer> transfers;
    bool in_use = false;
    bool retired = false;

    EmbeddingStatusSlot();
};

class EmbeddingStatusResources final {
public:
    explicit EmbeddingStatusResources(std::size_t capacity);
    EmbeddingStatusResources(const EmbeddingStatusResources&) = delete;
    EmbeddingStatusResources& operator=(const EmbeddingStatusResources&) = delete;

    [[nodiscard]] std::size_t acquire();
    void release(std::size_t slot, bool completion_proven) noexcept;
    [[nodiscard]] EmbeddingStatusSlot& get(std::size_t slot) noexcept;
    [[nodiscard]] const EmbeddingStatusSlot& get(std::size_t slot) const noexcept;

private:
    std::mutex mutex_;
    std::vector<EmbeddingStatusSlot> slots_;
};

class EmbeddingProgram final {
public:
    explicit EmbeddingProgram(tt::tt_metal::distributed::MeshDevice& device);
    EmbeddingProgram(const EmbeddingProgram&) = delete;
    EmbeddingProgram& operator=(const EmbeddingProgram&) = delete;

    void dispatch(
            tt::tt_metal::distributed::MeshCommandQueue& queue,
            const std::array<std::uint32_t, kEmbeddingRuntimeArgCount>& args);

private:
    tt::tt_metal::distributed::MeshWorkload workload_;
    tt::tt_metal::distributed::MeshCoordinateRange device_range_;
    tt::tt_metal::CoreCoord core_;
    tt::tt_metal::KernelHandle kernel_ = 0;
    tt::tt_metal::Program* program_ = nullptr;
};

void embedding_planes(
        tt::tt_metal::distributed::MeshDevice& device,
        EmbeddingProgram& program,
        EmbeddingStatusSlot& status,
        const EmbeddingNativeRequest& request,
        const NativeWorkspace& workspace,
        bool& any_submitted);

}  // namespace iom::ttnn_detail
