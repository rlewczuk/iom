#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <memory>

#include "backend/backend_conformance_model_reference.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

TEST_CASE("TinyLlama pinned model reference checkpoints") {
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
    REQUIRE(device != nullptr);
    iom_conformance::run_model_reference_conformance(*device);
}
