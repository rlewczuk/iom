#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace iom {

    enum class DataType {
        BOOL,

        I2, U2,
        I4, U4,
        I8, U8,
        I16, U16,
        I32, U32,
        I64, U64,

        F4_E2M1,
        F6_E2M3,
        F6_E3M2,

        F8_E4M3FN,
        F8_E5M2,
        F8_E8M0,

        F16,
        BF16,
        F32,
        F64,
    };

    enum class QuantizationFormat {
        // No data grouping, just plain numbers (of appropriate DataType)
        NONE,

        // Generic formats, if you actually implement them
        INT8_SYMMETRIC,
        INT8_ASYMMETRIC,
        INT4_SYMMETRIC,
        INT4_ASYMMETRIC,

        // OCP
        OCP_MXFP4,
        OCP_MXFP8_E4M3,
        OCP_MXFP8_E5M2,

        // NVIDIA
        NVIDIA_NVFP4,

        // GGML
        GGML_Q4_0,
        GGML_Q4_1,
        GGML_Q5_0,
        GGML_Q5_1,
        GGML_Q8_0,
        GGML_Q2_K,
        GGML_Q3_K,
        GGML_Q4_K,
        GGML_Q5_K,
        GGML_Q6_K,
        // ...

        // Vendor-specific formats
        TT_BFP2,
        TT_BFP2A,
        TT_BFP4,
        TT_BFP4A,
        TT_BFP8,
        TT_BFP8A,
    };

    enum class BackendKind : std::uint8_t {
        CPU,
        CUDA,
        ROCM,
        SYCL,
    };

    /**
     * Full tensor shape. Every materialized tensor, owner view, transformed
     * view result, and computed binary result shape carries a full rank
     * between two and eight inclusive; ranks below two and above eight are
     * rejected with std::invalid_argument at construction. The final two
     * dimensions are the tiled matrix axes; leading-dimension spans passed
     * to TensorView transforms are helper spans, never full shapes
     * themselves.
     */
    class TensorShape {
    public:
        explicit TensorShape(std::vector<std::size_t> dimensions);

        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t dimension(std::size_t index) const;
        [[nodiscard]] std::span<const std::size_t> dimensions() const noexcept;
        [[nodiscard]] std::size_t element_count() const;

        friend bool operator==(const TensorShape&, const TensorShape&) = default;

    private:
        std::vector<std::size_t> dimensions_;
    };

    /**
     * Full tensor specification: a full TensorShape of rank two through
     * eight, a leaf data type, and an optional quantization format.
     * validate() enforces the full-rank interval together with the leaf
     * type and quantization before any storage or metadata exists.
     */
    struct TensorSpec {
        static constexpr std::size_t TILE = 16;

        TensorShape shape;
        DataType data_type;
        QuantizationFormat quantization = QuantizationFormat::NONE;

        [[nodiscard]] TensorShape standard_padded_shape() const;
        [[nodiscard]] std::size_t logical_nbytes() const;
        [[nodiscard]] std::size_t tiled_storage_nbytes() const;
        void validate() const;

        friend bool operator==(const TensorSpec&, const TensorSpec&) = default;
    };

    namespace detail {

        // Checked element-slot address of full dense coordinates in the
        // standard 16x16 tiled layout (leading plane row-major, then tile
        // row, tile column, row in tile, column in tile). Not a public
        // tensor representation; shared by tests and the standard-layout
        // backends.
        [[nodiscard]] std::size_t standard_layout_slot(
            const TensorSpec& spec, std::span<const std::size_t> coordinates);

        // Exact bit width of every declared leaf encoding.
        [[nodiscard]] std::size_t leaf_bits(DataType type);

        // Checked element-slot address of (row, column) within one owner
        // plane of the standard 16x16 tiled layout. standard_layout_slot
        // derives the plane from dense coordinates; standard-layout
        // backends with strided views supply the owner plane directly.
        // Preconditions: spec validates and plane, row, and column address
        // existing planes and matrix elements.
        [[nodiscard]] std::size_t standard_plane_slot(
            const TensorSpec& spec, std::size_t plane,
            std::size_t row, std::size_t column);

        // Shared raw-workspace range validation machinery. Declared here so
        // RawWorkspaceView can grant it checked access to the owner range
        // without exposing workspace addresses for construction; defined
        // with the iom.hpp operation contract.
        class WorkspaceValidation;

    }  // namespace detail

    class Device;
    class DeviceOps;
    class Tensor;
    class RawWorkspace;

    /**
     * Deterministic byte capacity and alignment requirement for one raw
     * device workspace. Zero-byte requirements carry alignment one, and
     * every reported requirement is independent of free data-arena
     * capacity, queue occupancy, and completion state.
     */
    struct WorkspaceRequirements {
        std::size_t bytes = 0;
        std::size_t alignment = 1;

        friend bool operator==(
                const WorkspaceRequirements&,
                const WorkspaceRequirements&) = default;
    };

    /**
     * Non-owning checked byte range over one live, explicitly owned raw
     * workspace. `RawWorkspaceView{}` is the valid empty default; views are
     * copy-constructible and never retargetable, so every copy preserves
     * the exact owning `RawWorkspace`, its creating `Device`, and the
     * checked byte extent. Only a live owner can produce a view, and all
     * subrange arithmetic is checked against the owner range. No
     * constructor accepts an arbitrary pointer or backend handle.
     */
    class RawWorkspaceView {
    public:
        RawWorkspaceView() = default;
        RawWorkspaceView(const RawWorkspaceView&) = default;
        RawWorkspaceView(RawWorkspaceView&&) = delete;
        RawWorkspaceView& operator=(const RawWorkspaceView&) = delete;
        RawWorkspaceView& operator=(RawWorkspaceView&&) = delete;

        [[nodiscard]] bool empty() const noexcept {
            return owner_ == nullptr;
        }
        [[nodiscard]] std::size_t byte_size() const noexcept {
            return bytes_;
        }
        [[nodiscard]] std::size_t offset() const noexcept {
            return offset_;
        }
        // Checked range extents within the owning workspace: both derive
        // from values bounded at view construction.
        [[nodiscard]] std::size_t range_begin() const noexcept {
            return offset_;
        }
        [[nodiscard]] std::size_t range_end() const noexcept {
            return offset_ + bytes_;
        }
        [[nodiscard]] const RawWorkspace* owner_identity() const noexcept {
            return owner_;
        }
        [[nodiscard]] const Device& device() const;
        [[nodiscard]] BackendKind backend_kind() const;
        [[nodiscard]] std::uint32_t backend_device() const;

        // Owner-absolute checked subrange. The offset counts from the
        // owner base and must stay 32-byte aligned; the construction
        // invariants `offset <= owner_bytes` and
        // `bytes <= owner_bytes - offset` are checked before any view
        // value exists.
        [[nodiscard]] RawWorkspaceView subrange(
                std::size_t offset, std::size_t bytes) const;

        friend bool operator==(
                const RawWorkspaceView&,
                const RawWorkspaceView&) = default;

    private:
        friend class RawWorkspace;
        friend class DeviceOps;
        friend class detail::WorkspaceValidation;
        RawWorkspaceView(const RawWorkspace& owner, std::size_t offset,
                         std::size_t bytes);
        [[nodiscard]] void* range_address() const noexcept;

        const RawWorkspace* owner_ = nullptr;
        std::size_t offset_ = 0;
        std::size_t bytes_ = 0;
    };

    /**
     * Explicitly owned raw device workspace created through
     * Device::create_workspace. Non-copyable and non-movable so the owner
     * and its backing range never relocate or retarget while views or
     * accepted work exist; the creating Device must outlive the workspace.
     * The owner registers its exact identity with the Device for the whole
     * lifetime, which is what lets shared validation reject foreign and dead
     * owners without dereferencing them. No accelerator runtime type is
     * exposed.
     */
    class RawWorkspace {
    public:
        virtual ~RawWorkspace();
        RawWorkspace(const RawWorkspace&) = delete;
        RawWorkspace& operator=(const RawWorkspace&) = delete;
        RawWorkspace(RawWorkspace&&) = delete;
        RawWorkspace& operator=(RawWorkspace&&) = delete;

        [[nodiscard]] std::size_t byte_size() const noexcept {
            return bytes_;
        }
        [[nodiscard]] bool empty() const noexcept {
            return bytes_ == 0;
        }
        [[nodiscard]] const Device& device() const noexcept {
            return *device_;
        }
        [[nodiscard]] BackendKind backend_kind() const noexcept;
        [[nodiscard]] std::uint32_t backend_device() const noexcept;
        // Full-range view of this live owner.
        [[nodiscard]] RawWorkspaceView view() const;

    protected:
        RawWorkspace(const Device& device, std::size_t bytes);
        // Stable base address of the owned backing range; nullptr for an
        // empty (zero-byte) workspace. Backends report their arena
        // suballocation here.
        [[nodiscard]] virtual void* workspace_address() const noexcept;

    private:
        friend class RawWorkspaceView;
        const Device* device_;
        std::size_t bytes_;
    };
    class Tensor;

    /**
     * Non-owning tensor operand over one stable caller-owned storage owner:
     * a plane offset and plane strides count whole logical planes, never bytes
     * or elements. The final two dimensions are a logical matrix represented
     * by 16x16 tiles (native backends may use a different internal tile).
     * Leading transforms preserve independent offsets and strides. Every
     * transformed view result is a full shape within rank two through
     * eight; the leading-dimension spans reshape_leading and permute take
     * are helper spans, not themselves full shapes, and are bounded only
     * through the assembled result. Views are copyable but non-assignable;
     * operations snapshot metadata and never retain the view object or
     * expose broadcast zero strides.
     */
    class TensorView {
    public:
        TensorView(const TensorView&) = default;
        TensorView(TensorView&&) = default;
        TensorView& operator=(const TensorView&) = delete;
        TensorView& operator=(TensorView&&) = delete;

        [[nodiscard]] const TensorSpec& spec() const noexcept;
        [[nodiscard]] const Tensor* owner_identity() const noexcept;
        [[nodiscard]] const Device& device() const noexcept;
        [[nodiscard]] BackendKind backend_kind() const noexcept;
        [[nodiscard]] std::uint32_t backend_device() const noexcept;

        // Always the owner's storage handle; never retained past the owner.
        [[nodiscard]] void* native_handle() noexcept;
        [[nodiscard]] const void* native_handle() const noexcept;

        [[nodiscard]] std::size_t plane_offset() const noexcept;
        [[nodiscard]] std::span<const std::size_t> plane_strides() const noexcept;

        [[nodiscard]] TensorView slice(std::size_t dim, std::size_t first,
                                       std::size_t count, std::size_t step = 1) const;
        [[nodiscard]] TensorView select(std::size_t dim, std::size_t index) const;
        [[nodiscard]] TensorView permute(
            std::span<const std::size_t> leading_order) const;
        [[nodiscard]] TensorView reshape_leading(
            std::span<const std::size_t> leading_dimensions) const;

        // Synchronous host transfers of exactly spec().logical_nbytes().
        // Host access never waits on operation queues: callers wait for
        // outstanding writes before a host read, and for every outstanding
        // read or write before a host write or owner destruction. Queued
        // binary operations track all three owners through completion,
        // deduplicate exact aliases, snapshot metadata, and never allocate,
        // replace, or relocate storage.
        void copy_from_host(
                std::span<const std::byte> source,
                RawWorkspaceView workspace = {});
        void copy_to_host(
                std::span<std::byte> destination,
                RawWorkspaceView workspace = {}) const;

        // Pure deterministic host-transfer workspace requirements (leaf
        // 05). Each query validates the view specification with the same
        // checked arithmetic as the transfer itself and reports, without
        // any allocation, registration, leasing, or native effect:
        // CPU `{0, 1}`, and CUDA, ROCm, and SYCL
        // `{gpu_algorithm::compute_staging_size(logical_nbytes), 32}`.
        // The data-dependent BOOL byte check stays with the transfer
        // call: it is not a pure-query predicate.
        [[nodiscard]] WorkspaceRequirements
                copy_from_host_workspace_requirements() const;
        [[nodiscard]] WorkspaceRequirements
                copy_to_host_workspace_requirements() const;

    private:
        friend class Tensor;
        TensorView(Tensor& owner, TensorSpec spec, std::size_t plane_offset,
                   std::vector<std::size_t> plane_strides);

        Tensor* owner_;
        TensorSpec spec_;
        std::size_t plane_offset_ = 0;
        std::vector<std::size_t> plane_strides_;
    };

    /**
     * Materialized tensor owner created through a Device. Holds one stable
     * full-storage view whose full shape has rank two through eight;
     * construction validates the complete specification before any storage
     * is allocated or registered. Non-copyable and non-movable so the
     * owner and full-view addresses stay valid for asynchronous
     * operations; the creating Device must outlive the tensor.
     */
    class Tensor {
    public:
        virtual ~Tensor() = default;
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        Tensor(Tensor&&) = delete;
        Tensor& operator=(Tensor&&) = delete;

        [[nodiscard]] TensorView& view() noexcept;
        [[nodiscard]] const TensorView& view() const noexcept;

    protected:
        Tensor(TensorSpec spec, Device& device);

        // Computes the owner-specific host-transfer workspace policy from
        // checked logical bytes. This pure hook never allocates, registers,
        // leases, synchronizes, or otherwise affects backend state.
        [[nodiscard]] virtual WorkspaceRequirements
                host_transfer_workspace_requirements(
                        std::size_t checked_logical_nbytes) const = 0;

        [[nodiscard]] virtual void* storage_handle() noexcept = 0;
        virtual void region_from_host(
                const TensorView& destination,
                std::span<const std::byte> source,
                RawWorkspaceView workspace) = 0;
        virtual void region_to_host(
                const TensorView& source,
                std::span<std::byte> destination,
                RawWorkspaceView workspace) const = 0;

    private:
        friend class TensorView;
        Device* device_;
        TensorView full_view_;
    };

}
