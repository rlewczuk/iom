#include "iom/iom.hpp"

#include "iom_internal.hpp"
#include <algorithm>

namespace iom {

    using detail::UnsupportedOperation;

    void DeviceOps::validate_copy(
            const Device& device, const TensorView& source,
            const TensorView& destination) {
        validate_views(device, {&source, &destination});
        if (!(source.spec() == destination.spec())) {
            throw std::invalid_argument(
                    "copy views must have identical shape, leaf type, "
                    "and quantization");
        }
    }

    bool DeviceOps::identical_window(
            const TensorView& source, const TensorView& destination) {
        return source.native_handle() == destination.native_handle()
                && source.plane_offset() == destination.plane_offset()
                && std::equal(
                        source.plane_strides().begin(),
                        source.plane_strides().end(),
                        destination.plane_strides().begin(),
                        destination.plane_strides().end());
    }

    oid DeviceOps::copy_impl(
            const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::copy(
            const TensorView& source, TensorView& destination) noexcept {
        try {
            validate_copy(queue_device(), source, destination);
            return invoke(copy_impl(source, destination));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

}  // namespace iom
