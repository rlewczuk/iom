#pragma once

#include <cstddef>

namespace iom::sycl_detail {

    // Test-only observation points. They do not select a device or retain
    // runtime state; the backend owns the context independently of this seam.
    struct ContextCalls {
        void (*context_created)() = nullptr;
        void (*context_destroyed)() = nullptr;
    };

    extern ContextCalls context_calls;

    [[nodiscard]] std::size_t eligible_device_count();

}  // namespace iom::sycl_detail
