#include "copy.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace iom::ttnn_detail {

    namespace {

        // Number of logical planes a view addresses: the product of its
        // leading dimensions.
        std::size_t view_plane_count(const TensorView& view) {
            const std::span<const std::size_t> dimensions =
                    view.spec().shape.dimensions();
            std::size_t planes = 1;
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes *= dimensions[i];
            }
            return planes;
        }

        // Owner plane of the view's linear plane index, decomposed row-major
        // over the view's leading dimensions and mapped through the view's
        // plane offset and strides. View transforms keep every addressed
        // owner plane in bounds.
        std::size_t owner_plane_at(const TensorView& view, std::size_t index) {
            const std::span<const std::size_t> dimensions =
                    view.spec().shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::span<const std::size_t> strides =
                    view.plane_strides();
            std::size_t plane = view.plane_offset();
            for (std::size_t k = leading_rank; k-- > 0;) {
                plane += (index % dimensions[k]) * strides[k];
                index /= dimensions[k];
            }
            return plane;
        }

        // Every supported leaf type is byte-aligned, so one logical element
        // is one whole number of bytes.
        std::size_t element_bytes(DataType type) {
            const std::size_t bits = detail::leaf_bits(type);
            if (bits % 8 != 0) {
                throw std::logic_error(
                        "TTNN supported type is not byte-aligned");
            }
            return bits / 8;
        }

        // TTNN's host layout conversion views the buffer through the dtype's
        // C++ element type, so the raw bytes must arrive in a correctly
        // typed vector. This is a reinterpretation, never a conversion.
        template <typename T>
        tt::tt_metal::HostBuffer typed_buffer(std::vector<std::byte>& bytes) {
            std::vector<T> typed(bytes.size() / sizeof(T));
            std::memcpy(typed.data(), bytes.data(), bytes.size());
            return tt::tt_metal::HostBuffer(std::move(typed));
        }

        tt::tt_metal::HostBuffer make_host_buffer(
                tt::tt_metal::DataType dtype, std::vector<std::byte>& bytes) {
            switch (dtype) {
                case tt::tt_metal::DataType::BFLOAT16:
                    return typed_buffer<bfloat16>(bytes);
                case tt::tt_metal::DataType::FLOAT32:
                    return typed_buffer<float>(bytes);
                case tt::tt_metal::DataType::UINT32:
                    return typed_buffer<std::uint32_t>(bytes);
                case tt::tt_metal::DataType::INT32:
                    return typed_buffer<std::int32_t>(bytes);
                case tt::tt_metal::DataType::UINT16:
                    return typed_buffer<std::uint16_t>(bytes);
                case tt::tt_metal::DataType::UINT8:
                    return typed_buffer<std::uint8_t>(bytes);
                default:
                    throw std::logic_error(
                            "TTNN native dtype has no host element type");
            }
        }

        // Uploads one plane's logical row-major bytes into the plane's
        // TTNN-native tiled tensor, zero-filling native padding.
        void upload_plane(
                ttnn::Tensor& plane, const std::byte* source,
                std::size_t rows, std::size_t columns,
                std::size_t element_size) {
            const std::size_t padded_rows =
                    static_cast<std::size_t>(plane.padded_shape()[-2]);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(plane.padded_shape()[-1]);
            std::vector<std::byte> padded(
                    padded_rows * padded_columns * element_size,
                    std::byte{0});
            for (std::size_t row = 0; row < rows; ++row) {
                std::memcpy(
                        padded.data() + row * padded_columns * element_size,
                        source + row * columns * element_size,
                        columns * element_size);
            }
            ttnn::Tensor host_row_major(
                    make_host_buffer(plane.dtype(), padded),
                    plane.logical_shape(), plane.padded_shape(),
                    plane.dtype(), tt::tt_metal::Layout::ROW_MAJOR);
            const ttnn::Tensor host_tiled =
                    tt::tt_metal::to_layout(
                            host_row_major, tt::tt_metal::Layout::TILE);
            ttnn::copy_to_device(host_tiled, plane);
        }

        // Downloads one plane's logical row-major bytes from the plane's
        // TTNN-native tiled tensor. Native padding never reaches the output.
        void download_plane(
                tt::tt_metal::distributed::MeshDevice& device,
                const ttnn::Tensor& plane, std::byte* destination,
                std::size_t rows, std::size_t columns,
                std::size_t element_size) {
            ttnn::Tensor host_tiled =
                    ttnn::allocate_tensor_on_host(plane.tensor_spec(), &device);
            ttnn::copy_to_host(plane, host_tiled, /*blocking=*/true);
            const ttnn::Tensor host_row_major = tt::tt_metal::to_layout(
                    host_tiled, tt::tt_metal::Layout::ROW_MAJOR);
            const tt::tt_metal::HostBuffer buffer =
                    tt::tt_metal::host_buffer::get_host_buffer(
                            host_row_major.host_tensor());
            const auto bytes = buffer.view_bytes();
            const std::size_t padded_columns = static_cast<std::size_t>(
                    host_row_major.padded_shape()[-1]);
            for (std::size_t row = 0; row < rows; ++row) {
                std::memcpy(
                        destination + row * columns * element_size,
                        bytes.data() + row * padded_columns * element_size,
                        columns * element_size);
            }
        }

        std::size_t view_rows(const TensorView& view) {
            return view.spec().shape.dimension(
                    view.spec().shape.rank() - 2);
        }

        std::size_t view_columns(const TensorView& view) {
            return view.spec().shape.dimension(
                    view.spec().shape.rank() - 1);
        }

    }  // namespace

    void region_from_host(
            tt::tt_metal::distributed::MeshDevice& device,
            const TensorView& destination, ttnn::Tensor* planes,
            std::span<const std::byte> source) {
        const std::size_t rows = view_rows(destination);
        const std::size_t columns = view_columns(destination);
        const std::size_t element_size =
                element_bytes(destination.spec().data_type);
        const std::size_t plane_bytes = rows * columns * element_size;
        const std::size_t count = view_plane_count(destination);
        for (std::size_t index = 0; index < count; ++index) {
            upload_plane(
                    planes[owner_plane_at(destination, index)],
                    source.data() + index * plane_bytes, rows, columns,
                    element_size);
        }
        device.mesh_command_queue(0).finish();
    }

    void region_to_host(
            tt::tt_metal::distributed::MeshDevice& device,
            const TensorView& source, const ttnn::Tensor* planes,
            std::span<std::byte> destination) {
        // Zero first: unused tail bits read as zero and padding never
        // reaches the host buffer.
        std::fill(destination.begin(), destination.end(), std::byte{0});
        const std::size_t rows = view_rows(source);
        const std::size_t columns = view_columns(source);
        const std::size_t element_size =
                element_bytes(source.spec().data_type);
        const std::size_t plane_bytes = rows * columns * element_size;
        const std::size_t count = view_plane_count(source);
        for (std::size_t index = 0; index < count; ++index) {
            download_plane(
                    device, planes[owner_plane_at(source, index)],
                    reinterpret_cast<std::byte*>(destination.data())
                            + index * plane_bytes,
                    rows, columns, element_size);
        }
    }

    void copy_planes(
            const TensorView& source, const ttnn::Tensor* source_planes,
            const TensorView& destination, ttnn::Tensor* destination_planes) {
        const std::size_t count = view_plane_count(source);
        for (std::size_t index = 0; index < count; ++index) {
            ttnn::copy(
                    source_planes[owner_plane_at(source, index)],
                    destination_planes[owner_plane_at(destination, index)]);
        }
    }

}  // namespace iom::ttnn_detail
