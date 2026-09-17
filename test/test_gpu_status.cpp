#include "../src/shared/event_ring.hpp"
#include <doctest/doctest.h>
#include <cstdint>
#include <stdexcept>

namespace {
struct Policy {
    using context_type = int;
    using event_type = int*;
    static inline bool fail_sync = false;
    static void activate(int) {}
    static void check_queue_event_fault() {}
    static void check_acquire_event_fault() {}
    static void create_event(int** event) { *event = new int(0); }
    static void destroy_event_noexcept(int* event) noexcept { delete event; }
    static void synchronize_event(int*) {
        if (fail_sync) throw std::runtime_error("event failure");
    }
    static void synchronize_event_noexcept(int*) noexcept {}
};
struct Provider final : iom::detail::QueueResourceProvider {
    std::uint32_t status = 0;
    std::byte device[512]{};
    std::size_t queue_slot_count() const noexcept override { return 1; }
    iom::detail::QueueResourceLease reserve_queue_resources() override {
        return make_lease(*this, 0, 1, device,
                std::make_unique<std::byte[]>(512), &status);
    }
    void release_queue_resources(std::size_t, std::size_t) noexcept override {}
    void retain_unknown_lease(std::function<bool()>, std::function<void()>) override {}
    void reclaim_retained_leases() override {}
};
}  // namespace

TEST_CASE("GPU embedding status survives metadata slot reuse") {
    using Ring = iom::detail::EventRingState<Policy>;
    for (bool shutdown : {false, true}) {
        for (std::uint32_t first : {0u, 1u}) {
            Provider provider;
            iom::detail::MetadataSlotPool pool(provider.reserve_queue_resources());
            auto ring = std::make_shared<Ring>(0, pool, 1);
            auto a = ring->acquire();
            const auto slot = pool.acquire();
            a->attach_metadata_slot(slot);
            a->attach_status_cell(pool.status_data(slot));
            provider.status = first;
            ring->mark_event_recorded(*a);
            if (!shutdown) ring->on_worker_complete(*a);
            ring->on_worker_destroy(*a);
            // Force the exact release -> reuse/write -> consume ordering.
            auto b = ring->acquire();
            const auto reused = pool.acquire();
            REQUIRE_EQ(reused, slot);
            b->attach_metadata_slot(reused);
            provider.status = 1u - first;
            const auto retained = a->status_word();
            REQUIRE(retained.has_value());
            CHECK_EQ(*retained, first);
            CHECK_EQ(a->status_word(), retained);
            ring->mark_stream_drained(*b);
            ring->on_worker_destroy(*b);
        }
    }
}
