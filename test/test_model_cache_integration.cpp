#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <memory>

#include "backend/backend_conformance_model_cache.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

TEST_CASE("TinyLlama pinned causal cache reference conformance") {
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
    REQUIRE(device != nullptr);
    iom_conformance::run_model_cache_reference_conformance(*device);
}
