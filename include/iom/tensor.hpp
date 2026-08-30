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

        // Tenstorrent
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
        TTNN,
    };

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

        // Checked element-slot address in the standard 16x16 tiled layout
        // (leading plane row-major, then tile row, tile column, row in tile,
        // column in tile). Not a public tensor representation; shared by
        // tests and the standard-layout backends.
        [[nodiscard]] std::size_t standard_layout_slot(
            const TensorSpec& spec, std::span<const std::size_t> coordinates);

    }  // namespace detail

    class Device;
    class Tensor;

    /**
     * Non-owning tensor operand over one owner's storage: a plane offset and
     * plane strides counting whole logical planes, never bytes or elements.
     * The final two dimensions are the tiled matrix and cannot be
     * transformed. Copyable so operations can receive views by value, but
     * never assignable so a view can never be retargeted.
     */
    class TensorView {
    public:
        TensorView(const TensorView&) = default;
        TensorView(TensorView&&) = default;
        TensorView& operator=(const TensorView&) = delete;
        TensorView& operator=(TensorView&&) = delete;

        [[nodiscard]] const TensorSpec& spec() const noexcept;
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
        // read or write before a host write or owner destruction.
        void copy_from_host(std::span<const std::byte> source);
        void copy_to_host(std::span<std::byte> destination) const;

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
     * full-storage view. Non-copyable and non-movable so the owner and
     * full-view addresses stay valid for asynchronous operations; the
     * creating Device must outlive the tensor.
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

        [[nodiscard]] virtual void* storage_handle() noexcept = 0;
        virtual void region_from_host(const TensorView& destination,
                                      std::span<const std::byte> source) = 0;
        virtual void region_to_host(const TensorView& source,
                                    std::span<std::byte> destination) const = 0;

    private:
        friend class TensorView;
        Device* device_;
        TensorView full_view_;
    };

}
