#include <doctest/doctest.h>

#include <condition_variable>
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <thread>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/iom.hpp"
#include "iom/token_selection.hpp"
#include "model_loading_fixture.hpp"

namespace {

class HeapAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t bytes) override {
        return ::operator new(
                std::max<std::size_t>(bytes, 1), std::align_val_t{32});
    }

    void free(void* address) override {
        ::operator delete(address, std::align_val_t{32});
    }

    void reset() override {}
};

class ManualQueue final : public iom::DeviceOps {
public:
    ManualQueue() = default;
    explicit ManualQueue(const iom::Device& device) : iom::DeviceOps(device) {}

    using iom::DeviceOps::commit_failure;
    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;
    using iom::DeviceOps::submit;

    [[nodiscard]] iom::oid probe() {
        return submit([](std::uint64_t) {});
    }

    void fail(iom::oid token, const char* message) {
        commit_failure(
                sequence(token),
                std::make_exception_ptr(std::runtime_error(message)));
        complete(sequence(token));
    }

    static std::uint64_t sequence(iom::oid token) noexcept {
        constexpr std::uint64_t kMask = (std::uint64_t{1} << 55) - 1;
        return static_cast<std::uint64_t>(token) & kMask;
    }
};
class DeferredCopyQueue final : public iom::DeviceOps {
public:
    explicit DeferredCopyQueue(const iom::Device& device)
        : iom::DeviceOps(device) {}

    using iom::DeviceOps::commit_failure;
    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;
    using iom::DeviceOps::submit;

    struct Record {
        std::uint64_t sequence;
        const iom::TensorView* source;
        iom::TensorView* destination;
        bool released;
    };

    [[nodiscard]] iom::oid probe() {
        return submit([this](std::uint64_t sequence) {
            records_.push_back({sequence, nullptr, nullptr, false});
        });
    }

    void release(iom::oid token) {
        Record* record = find(sequence(token));
        if (record == nullptr || record->source == nullptr
                || record->destination == nullptr) {
            throw std::logic_error("deferred copy record is unavailable");
        }
        std::vector<std::byte> bytes(record->source->spec().logical_nbytes());
        record->source->copy_to_host(bytes);
        record->destination->copy_from_host(bytes);
        record->released = true;
        complete(record->sequence);
    }

    void fail(iom::oid token, const char* message) {
        commit_failure(
                sequence(token),
                std::make_exception_ptr(std::runtime_error(message)));
        complete(sequence(token));
    }

    [[nodiscard]] static std::uint64_t sequence(iom::oid token) noexcept {
        constexpr std::uint64_t kMask = (std::uint64_t{1} << 55) - 1;
        return static_cast<std::uint64_t>(token) & kMask;
    }

protected:
    iom::oid copy_impl(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        return submit([this, &source, &destination](
                              std::uint64_t sequence) {
            records_.push_back(
                    {sequence, &source, &destination, false});
        });
    }

private:
    [[nodiscard]] Record* find(std::uint64_t sequence) noexcept {
        for (Record& record : records_) {
            if (record.sequence == sequence) return &record;
        }
        return nullptr;
    }

    std::vector<Record> records_;
};


struct CpuFixture {
    HeapAllocator allocator;
    std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
};

[[nodiscard]] iom::TensorSpec logits_spec(std::size_t vocabulary) {
    return iom::TensorSpec{
            iom::TensorShape{{1, vocabulary}}, iom::DataType::BF16};
}

[[nodiscard]] std::vector<std::byte> encode_bf16(
        std::span<const std::uint16_t> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        bytes[2 * index] = static_cast<std::byte>(values[index] & 0xffu);
        bytes[2 * index + 1] =
                static_cast<std::byte>((values[index] >> 8) & 0xffu);
    }
    return bytes;
}

[[nodiscard]] std::vector<std::byte> encode_bf16(
        const std::vector<std::uint16_t>& values) {
    return encode_bf16(std::span<const std::uint16_t>(values));
}

void poison_padding(iom::Tensor& tensor) {
    const iom::TensorSpec spec = tensor.view().spec();
    const iom::TensorShape padded = spec.standard_padded_shape();
    auto* storage = static_cast<std::byte*>(tensor.view().native_handle());
    for (std::size_t row = 0; row < padded.dimension(0); ++row) {
        for (std::size_t column = 0; column < padded.dimension(1);
             ++column) {
            if (row < 1 && column < spec.shape.dimension(1)) continue;
            const std::size_t slot = iom::detail::standard_plane_slot(
                    spec, 0, row, column);
            storage[2 * slot] = std::byte{0x80};
            storage[2 * slot + 1] = std::byte{0x7f};
        }
    }
}
[[nodiscard]] std::vector<std::byte> snapshot_storage(
        const iom::Tensor& tensor) {
    const iom::TensorView view = tensor.view();
    const auto* storage = static_cast<const std::byte*>(view.native_handle());
    const std::size_t bytes = view.spec().tiled_storage_nbytes();
    return std::vector<std::byte>(storage, storage + bytes);
}


struct PreparedLogits {
    std::unique_ptr<iom::Tensor> source;
    std::unique_ptr<iom::Tensor> logits;
    std::unique_ptr<iom::DeviceOps> queue;
    iom::oid producer = 0;
};

[[nodiscard]] PreparedLogits prepare_cpu_logits(
        iom::Device& device, std::span<const std::uint16_t> values) {
    const iom::TensorSpec spec = logits_spec(values.size());
    PreparedLogits prepared{
            device.create_tensor(spec), device.create_tensor(spec),
            device.create_ops(), 0};
    const std::vector<std::byte> encoded = encode_bf16(values);
    prepared.source->view().copy_from_host(encoded);
    poison_padding(*prepared.logits);
    prepared.producer = prepared.queue->copy(
            prepared.source->view(), prepared.logits->view());
    REQUIRE(iom::oid_is_token(prepared.producer));
    return prepared;
}

void check_scratch_contents(
        std::span<const std::byte> actual, std::span<const std::byte> expected,
        std::byte tail) {
    REQUIRE_GE(actual.size(), expected.size());
    CHECK(std::equal(expected.begin(), expected.end(), actual.begin()));
    for (std::size_t index = expected.size(); index < actual.size(); ++index) {
        CHECK_EQ(actual[index], tail);
    }
}

}  // namespace

TEST_CASE("greedy token selection handles boundaries, ties, and padding") {
    struct Case {
        std::vector<std::uint16_t> values;
        std::size_t expected;
    };
    std::vector<Case> cases;
    cases.push_back({{0x3f80u}, 0});
    std::vector<std::uint16_t> first(17, 0xbf80u);
    first[0] = 0x7f7fu;
    cases.push_back({std::move(first), 0});

    std::vector<std::uint16_t> all_negative(15, 0xbf80u);
    all_negative[7] = 0xc000u;
    cases.push_back({std::move(all_negative), 0});

    std::vector<std::uint16_t> last(16, 0x3f80u);
    last[15] = 0x4000u;
    cases.push_back({std::move(last), 15});

    std::vector<std::uint16_t> finite_edges(17, 0xbf80u);
    finite_edges[0] = 0x0001u;  // positive finite subnormal
    finite_edges[1] = 0x8001u;  // negative finite subnormal
    finite_edges[2] = 0x7f7fu;  // largest finite BF16, tied at the interior
    finite_edges[3] = 0x7f7fu;  // exact tie keeps ID 2
    finite_edges[4] = 0xff7fu;  // largest-magnitude finite negative value
    finite_edges[5] = 0x8000u;  // -0
    finite_edges[6] = 0x0000u;  // +0, equal to -0
    cases.push_back({std::move(finite_edges), 2});

    for (const Case& test : cases) {
        CpuFixture fixture;
        PreparedLogits prepared = prepare_cpu_logits(
                *fixture.device, std::span<const std::uint16_t>(test.values));
        iom::GreedyTokenSelector selector;
        const auto requirements = selector.scratch_requirements(
                prepared.logits->view(), test.values.size());
        CHECK_EQ(requirements.host_bytes, test.values.size() * 2);
        CHECK_EQ(requirements.device, iom::WorkspaceRequirements{0, 1});

        const std::vector<std::byte> expected = encode_bf16(test.values);
        constexpr std::byte kTail{0xA5};
        std::vector<std::byte> host(
                requirements.host_bytes + 9, kTail);
        const iom::Tensor* logits_owner =
                prepared.logits->view().owner_identity();
        const std::vector<std::size_t> history{11, 12, 13};
        const std::vector<std::size_t> history_before = history;
        CHECK_EQ(
                selector.select(
                        *prepared.queue, prepared.logits->view(),
                        test.values.size(), prepared.producer, history,
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), {}}),
                test.expected);
        check_scratch_contents(host, expected, kTail);
        CHECK(history == history_before);
        CHECK_EQ(
                prepared.logits->view().owner_identity(), logits_owner);
    }
}

TEST_CASE("greedy token selection rejects every logical nonfinite value") {
    for (const std::uint16_t nonfinite : {0x7f80u, 0xff80u, 0x7fc1u}) {
        CpuFixture fixture;
        std::vector<std::uint16_t> values(15, 0x3f80u);
        values[0] = nonfinite;
        PreparedLogits prepared = prepare_cpu_logits(
                *fixture.device, std::span<const std::uint16_t>(values));
        iom::GreedyTokenSelector selector;
        const auto requirements = selector.scratch_requirements(
                prepared.logits->view(), values.size());
        std::vector<std::byte> host(requirements.host_bytes, std::byte{0xA5});
        CHECK_THROWS_AS(
                selector.select(
                        *prepared.queue, prepared.logits->view(), values.size(),
                        prepared.producer, {},
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), {}}),
                std::runtime_error);

        values[0] = 0x3f80u;
        values[14] = nonfinite;
        PreparedLogits losing = prepare_cpu_logits(
                *fixture.device, std::span<const std::uint16_t>(values));
        CHECK_THROWS_AS(
                selector.select(
                        *losing.queue, losing.logits->view(), values.size(),
                        losing.producer, {},
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), {}}),
                std::runtime_error);
    }
}

TEST_CASE("greedy token selection validates admission before producer wait") {
    CpuFixture fixture;
    const iom::TensorSpec spec = logits_spec(16);
    auto source = fixture.device->create_tensor(spec);
    auto logits = fixture.device->create_tensor(spec);
    const std::vector<std::uint16_t> values(16, 0x3f80u);
    const std::vector<std::byte> encoded = encode_bf16(values);
    source->view().copy_from_host(encoded);
    auto queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid producer = queue->probe();
    iom::GreedyTokenSelector selector;
    std::vector<std::byte> small(encoded.size() - 1, std::byte{0xA5});
    std::vector<std::byte> host(encoded.size(), std::byte{0xA5});

    CHECK_THROWS_AS(
            selector.select(
                    *queue, logits->view(), values.size(), producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(small), {}}),
            std::invalid_argument);
    queue->complete(ManualQueue::sequence(producer));
    queue.reset();
    auto shape_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid shape_producer = shape_queue->probe();
    auto wrong_shape = fixture.device->create_tensor(iom::TensorSpec{
            iom::TensorShape{{2, 16}}, iom::DataType::BF16});
    CHECK_THROWS_AS(
            selector.select(
                    *shape_queue, wrong_shape->view(), values.size(),
                    shape_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    shape_queue->complete(ManualQueue::sequence(shape_producer));
    shape_queue.reset();

    auto range_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid range_producer = range_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *range_queue, logits->view(), 15, range_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    range_queue->complete(ManualQueue::sequence(range_producer));
    range_queue.reset();

    auto zero_range_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid zero_range_producer = zero_range_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *zero_range_queue, logits->view(), 0,
                    zero_range_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    zero_range_queue->complete(
            ManualQueue::sequence(zero_range_producer));
    zero_range_queue.reset();

    auto invalid_oid_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid own_oid = invalid_oid_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *invalid_oid_queue, logits->view(), values.size(), 0, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    invalid_oid_queue->complete(ManualQueue::sequence(own_oid));
    invalid_oid_queue.reset();

    auto future_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid submitted = future_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *future_queue, logits->view(), values.size(),
                    submitted + 1, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    future_queue->complete(ManualQueue::sequence(submitted));
    future_queue.reset();

    auto target_queue = std::make_unique<ManualQueue>(*fixture.device);
    auto other_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid target_producer = target_queue->probe();
    const iom::oid same_device_foreign_producer = other_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *target_queue, logits->view(), values.size(),
                    same_device_foreign_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    target_queue->complete(ManualQueue::sequence(target_producer));
    other_queue->complete(
            ManualQueue::sequence(same_device_foreign_producer));
    target_queue.reset();
    other_queue.reset();

    auto skipped_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid first_submitted = skipped_queue->probe();
    skipped_queue->complete(ManualQueue::sequence(first_submitted));
    skipped_queue->seek_next_sequence(4);
    const iom::oid later = skipped_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *skipped_queue, logits->view(), values.size(), later - 1,
                    {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    skipped_queue->complete(ManualQueue::sequence(later));
    skipped_queue.reset();


    auto foreign_device = iom::make_cpu_device(fixture.allocator);
    auto foreign_logits = foreign_device->create_tensor(spec);
    auto foreign_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid foreign_producer = foreign_queue->probe();
    CHECK_THROWS_AS(
            selector.select(
                    *foreign_queue, foreign_logits->view(), values.size(),
                    foreign_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    foreign_queue->complete(ManualQueue::sequence(foreign_producer));
    foreign_queue.reset();

    auto overlap_queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid overlap_producer = overlap_queue->probe();
    auto* overlapping = static_cast<std::byte*>(logits->view().native_handle());
    CHECK_THROWS_AS(
            selector.select(
                    *overlap_queue, logits->view(), values.size(),
                    overlap_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(overlapping, encoded.size()),
                            {}}),
            std::invalid_argument);
    overlap_queue->complete(ManualQueue::sequence(overlap_producer));
    overlap_queue.reset();

    ManualQueue device_less;
    CHECK_THROWS_AS(
            selector.select(
                    device_less, logits->view(), values.size(), 1, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
}

TEST_CASE("greedy token selection preserves repeated producer failures") {
    CpuFixture fixture;
    const std::vector<std::uint16_t> values(1, 0x3f80u);
    auto logits = fixture.device->create_tensor(logits_spec(values.size()));
    iom::GreedyTokenSelector selector;
    auto queue = std::make_unique<ManualQueue>(*fixture.device);
    const iom::oid producer = queue->probe();
    queue->fail(producer, "retained producer failure");
    std::vector<std::byte> host(2, std::byte{0xA5});
    for (int attempt = 0; attempt < 2; ++attempt) {
        CHECK_THROWS_WITH(
                selector.select(
                        *queue, logits->view(), values.size(), producer, {},
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), {}}),
                "retained producer failure");
        CHECK_EQ(host[0], std::byte{0xA5});
        CHECK_EQ(host[1], std::byte{0xA5});
    }
}

TEST_CASE("greedy token selection propagates transfer failures") {
    iom_model_loading::FakeDevice device;
    auto logits = device.create_tensor(logits_spec(1));
    ManualQueue queue(device);
    const iom::oid producer = queue.probe();
    queue.complete(ManualQueue::sequence(producer));
    iom::GreedyTokenSelector selector;
    std::vector<std::byte> host(2, std::byte{0xA5});
    CHECK_THROWS_AS(
            selector.select(
                    queue, logits->view(), 1, producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::logic_error);
    CHECK_EQ(host[0], std::byte{0xA5});
    CHECK_EQ(host[1], std::byte{0xA5});
}

TEST_CASE("greedy token selection rejects invalid shape and checked overflow") {
    CpuFixture fixture;
    iom::GreedyTokenSelector selector;
    auto wrong_dtype = fixture.device->create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, 16}}, iom::DataType::F32});
    CHECK_THROWS_AS(
            selector.scratch_requirements(wrong_dtype->view(), 16),
            std::invalid_argument);

    iom_model_loading::FakeDevice bounded;
    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    auto overflowing = bounded.create_tensor(iom::TensorSpec{
            iom::TensorShape{{1, huge}}, iom::DataType::BF16});
    CHECK_THROWS_AS(
            selector.scratch_requirements(overflowing->view(), huge),
            std::overflow_error);
}

TEST_CASE("greedy token selection validates positive device scratch") {
    iom_model_loading::FakeDevice device;
    device.set_workspace_policy({1, 1});
    auto logits = device.create_tensor(logits_spec(1));
    ManualQueue queue(device);
    const iom::oid producer = queue.probe();
    iom::GreedyTokenSelector selector;
    std::vector<std::byte> host(2, std::byte{0xA5});

    CHECK(selector.scratch_requirements(logits->view(), 1).device
          == iom::WorkspaceRequirements{2, 1});
    CHECK_THROWS_AS(
            selector.select(
                    queue, logits->view(), 1, producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), {}}),
            std::invalid_argument);
    auto undersized = std::make_unique<iom_model_loading::BoundedWorkspace>(
            device, 1, reinterpret_cast<void*>(0x1000));
    CHECK_THROWS_AS(
            selector.select(
                    queue, logits->view(), 1, producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), undersized->view()}),
            std::invalid_argument);

    auto overlapping = std::make_unique<iom_model_loading::BoundedWorkspace>(
            device, 2, logits->view().native_handle());
    CHECK_THROWS_AS(
            selector.select(
                    queue, logits->view(), 1, producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), overlapping->view()}),
            std::invalid_argument);

    queue.complete(ManualQueue::sequence(producer));
}
TEST_CASE(
        "token selection readiness and lifetime rejects invalid producer OIDs before effects") {
    CpuFixture fixture;
    const std::vector<std::uint16_t> values(17, 0x3f80u);
    const std::vector<std::uint16_t> stale(17, 0xbf80u);
    auto source = fixture.device->create_tensor(logits_spec(values.size()));
    auto logits = fixture.device->create_tensor(logits_spec(values.size()));
    source->view().copy_from_host(encode_bf16(values));
    std::vector<std::uint16_t> stale_values = stale;
    stale_values[0] = 0x4000u;
    logits->view().copy_from_host(encode_bf16(stale_values));
    poison_padding(*logits);

    iom::TensorView source_view = source->view();
    iom::TensorView logits_view = logits->view();
    DeferredCopyQueue queue(*fixture.device);
    const iom::oid producer = queue.copy(source_view, logits_view);
    REQUIRE(iom::oid_is_token(producer));

    iom::GreedyTokenSelector selector;
    constexpr std::byte kTail{0xA5};
    std::vector<std::byte> host(
            values.size() * sizeof(std::uint16_t) + 7, kTail);
    auto workspace = fixture.device->create_workspace(0);
    const iom::RawWorkspaceView device_scratch = workspace->view();
    const iom::RawWorkspace* workspace_identity =
            device_scratch.owner_identity();
    const iom::Tensor* logits_owner = logits_view.owner_identity();
    const std::vector<std::size_t> history{31, 32, 33};
    const std::vector<std::size_t> history_before = history;
    const std::vector<std::byte> source_before = snapshot_storage(*source);
    const std::vector<std::byte> logits_before = snapshot_storage(*logits);

    const auto reject = [&](iom::DeviceOps& candidate, iom::oid token) {
        const std::vector<std::byte> host_before = host;
        CHECK_THROWS_AS(
                selector.select(
                        candidate, logits_view, values.size(), token, history,
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), device_scratch}),
                std::invalid_argument);
        CHECK(host == host_before);
        CHECK(snapshot_storage(*source) == source_before);
        CHECK(snapshot_storage(*logits) == logits_before);
        CHECK(history == history_before);
        CHECK_EQ(logits_view.owner_identity(), logits_owner);
        CHECK_EQ(device_scratch.owner_identity(), workspace_identity);
    };

    // Zero and negative OIDs are rejected before queue readiness is observed.
    reject(queue, 0);
    reject(queue, -1);

    // Both future and otherwise unsubmitted positive sequences are invalid.
    reject(queue, producer + 1);
    reject(queue, producer + 100);

    DeferredCopyQueue foreign_queue(*fixture.device);
    const iom::oid foreign = foreign_queue.probe();
    reject(queue, foreign);

    DeferredCopyQueue skipped_queue(*fixture.device);
    const iom::oid first = skipped_queue.probe();
    skipped_queue.complete(DeferredCopyQueue::sequence(first));
    skipped_queue.seek_next_sequence(4);
    const iom::oid later = skipped_queue.probe();
    reject(skipped_queue, later - 1);

    // Invalid admission did not consume or complete the real producer.
    queue.release(producer);
    const std::vector<std::byte> expected = encode_bf16(values);
    CHECK_EQ(
            selector.select(
                    queue, logits_view, values.size(), producer, history,
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            0);
    check_scratch_contents(host, expected, kTail);

    // The same queue remains usable after every rejected OID.
    const iom::oid second = queue.copy(source_view, logits_view);
    REQUIRE(iom::oid_is_token(second));
    queue.release(second);
    CHECK_EQ(
            selector.select(
                    queue, logits_view, values.size(), second, history,
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            0);

    foreign_queue.complete(DeferredCopyQueue::sequence(foreign));
    skipped_queue.complete(DeferredCopyQueue::sequence(later));
}

TEST_CASE(
        "token selection readiness and lifetime waits for deferred producer and releases borrowed inputs") {
    CpuFixture fixture;
    iom::GreedyTokenSelector selector;
    auto workspace = fixture.device->create_workspace(0);
    const iom::RawWorkspaceView device_scratch = workspace->view();
    const iom::RawWorkspace* workspace_identity =
            device_scratch.owner_identity();
    constexpr std::byte kTail{0xA5};
    std::vector<std::byte> host(17 * sizeof(std::uint16_t) + 9, kTail);

    {
        std::vector<std::uint16_t> values(17, 0xbf80u);
        values[12] = 0x4000u;
        std::vector<std::uint16_t> stale(17, 0x3f80u);
        stale[0] = 0x4000u;
        auto source = fixture.device->create_tensor(logits_spec(values.size()));
        auto logits = fixture.device->create_tensor(logits_spec(values.size()));
        source->view().copy_from_host(encode_bf16(values));
        logits->view().copy_from_host(encode_bf16(stale));
        poison_padding(*logits);
        iom::TensorView source_view = source->view();
        iom::TensorView logits_view = logits->view();
        DeferredCopyQueue queue(*fixture.device);
        const iom::oid producer = queue.copy(source_view, logits_view);
        REQUIRE(iom::oid_is_token(producer));

        const iom::Tensor* logits_owner = logits_view.owner_identity();
        const void* logits_native = logits_view.native_handle();
        const std::vector<std::size_t> history{41, 42};
        const std::vector<std::size_t> history_before = history;
        const std::vector<std::byte> source_before =
                snapshot_storage(*source);
        const std::vector<std::byte> expected = encode_bf16(values);
        std::size_t selected = std::numeric_limits<std::size_t>::max();
        std::exception_ptr selection_failure;
        bool started = false;
        std::mutex start_mutex;
        std::condition_variable start_cv;

        std::thread selection_thread([&] {
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                started = true;
            }
            start_cv.notify_one();
            try {
                selected = selector.select(
                        queue, logits_view, values.size(), producer, history,
                        iom::TokenSelectorScratch{
                                std::span<std::byte>(host), device_scratch});
            } catch (...) {
                selection_failure = std::current_exception();
            }
        });
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            start_cv.wait(lock, [&] { return started; });
        }

        // The selector call has started before the producer is released.
        queue.release(producer);
        const std::vector<std::byte> destination_after_release =
                snapshot_storage(*logits);
        selection_thread.join();

        CHECK_FALSE(selection_failure);
        CHECK_EQ(selected, 12);
        check_scratch_contents(host, expected, kTail);
        CHECK(snapshot_storage(*source) == source_before);
        CHECK(snapshot_storage(*logits) == destination_after_release);
        CHECK(history == history_before);
        CHECK_EQ(logits_view.owner_identity(), logits_owner);
        CHECK_EQ(logits_view.native_handle(), logits_native);
        CHECK_EQ(device_scratch.owner_identity(), workspace_identity);
    }

    // The caller can destroy every prior view/owner/history immediately and
    // use the same selector and scratch again.
    const std::vector<std::uint16_t> next_values{
            0x3f80u, 0x4000u, 0x3f80u};
    PreparedLogits next = prepare_cpu_logits(
            *fixture.device, std::span<const std::uint16_t>(next_values));
    std::fill(host.begin(), host.end(), kTail);
    CHECK_EQ(
            selector.select(
                    *next.queue, next.logits->view(), next_values.size(),
                    next.producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            1);
    check_scratch_contents(host, encode_bf16(next_values), kTail);
}

TEST_CASE(
        "token selection readiness and lifetime retains failures and reuses scratch") {
    CpuFixture fixture;
    iom::GreedyTokenSelector selector;
    auto workspace = fixture.device->create_workspace(0);
    const iom::RawWorkspaceView device_scratch = workspace->view();
    const iom::RawWorkspace* workspace_identity =
            device_scratch.owner_identity();
    constexpr std::byte kTail{0xA5};
    std::vector<std::byte> host(17 * sizeof(std::uint16_t) + 5, kTail);

    {
        const std::vector<std::uint16_t> values(3, 0x3f80u);
        auto source = fixture.device->create_tensor(logits_spec(values.size()));
        auto logits = fixture.device->create_tensor(logits_spec(values.size()));
        source->view().copy_from_host(encode_bf16(values));
        poison_padding(*logits);
        iom::TensorView source_view = source->view();
        iom::TensorView logits_view = logits->view();
        DeferredCopyQueue queue(*fixture.device);
        const iom::oid producer = queue.copy(source_view, logits_view);
        REQUIRE(iom::oid_is_token(producer));
        queue.fail(producer, "retained deferred producer failure");
        const std::vector<std::byte> logits_before =
                snapshot_storage(*logits);
        const std::vector<std::byte> host_before = host;
        for (int attempt = 0; attempt < 2; ++attempt) {
            CHECK_THROWS_WITH(
                    selector.select(
                            queue, logits_view, values.size(), producer, {},
                            iom::TokenSelectorScratch{
                                    std::span<std::byte>(host),
                                    device_scratch}),
                    "retained deferred producer failure");
            CHECK(host == host_before);
            CHECK(snapshot_storage(*logits) == logits_before);
            CHECK_THROWS_WITH(
                    queue.wait(producer),
                    "retained deferred producer failure");
        }
    }

    // A separate valid producer proves that a retained failure does not poison
    // the caller's reusable scratch.
    const std::vector<std::uint16_t> successful_values = [] {
        std::vector<std::uint16_t> values(15, 0x3f80u);
        values[14] = 0x4000u;
        return values;
    }();
    PreparedLogits successful = prepare_cpu_logits(
            *fixture.device,
            std::span<const std::uint16_t>(successful_values));
    std::fill(host.begin(), host.end(), kTail);
    CHECK_EQ(
            selector.select(
                    *successful.queue, successful.logits->view(),
                    successful_values.size(), successful.producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            14);
    check_scratch_contents(host, encode_bf16(successful_values), kTail);
    CHECK_EQ(device_scratch.owner_identity(), workspace_identity);

    // A nonfinite scan writes only the permitted host scratch and leaves it
    // reusable for the next independent call.
    std::vector<std::uint16_t> nonfinite_values(5, 0x3f80u);
    nonfinite_values[2] = 0x7f80u;
    PreparedLogits nonfinite = prepare_cpu_logits(
            *fixture.device,
            std::span<const std::uint16_t>(nonfinite_values));
    std::fill(host.begin(), host.end(), kTail);
    CHECK_THROWS_AS(
            selector.select(
                    *nonfinite.queue, nonfinite.logits->view(),
                    nonfinite_values.size(), nonfinite.producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            std::runtime_error);
    CHECK_EQ(device_scratch.owner_identity(), workspace_identity);

    // This completed queue has a synchronous transfer failure, not a retained
    // producer failure: its OID remains waitable and successful.
    iom_model_loading::FakeDevice transfer_device;
    auto transfer_logits = transfer_device.create_tensor(logits_spec(1));
    ManualQueue transfer_queue(transfer_device);
    const iom::oid transfer_producer = transfer_queue.probe();
    transfer_queue.complete(ManualQueue::sequence(transfer_producer));
    std::fill(host.begin(), host.end(), kTail);
    const std::vector<std::byte> transfer_host_before = host;
    CHECK_THROWS_AS(
            selector.select(
                    transfer_queue, transfer_logits->view(), 1,
                    transfer_producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            std::logic_error);
    CHECK(host == transfer_host_before);
    CHECK_NOTHROW(transfer_queue.wait(transfer_producer));
    CHECK_EQ(device_scratch.owner_identity(), workspace_identity);

    const std::vector<std::uint16_t> final_values{0x4000u};
    PreparedLogits final = prepare_cpu_logits(
            *fixture.device,
            std::span<const std::uint16_t>(final_values));
    std::fill(host.begin(), host.end(), kTail);
    CHECK_EQ(
            selector.select(
                    *final.queue, final.logits->view(), final_values.size(),
                    final.producer, {},
                    iom::TokenSelectorScratch{
                            std::span<std::byte>(host), device_scratch}),
            0);
    check_scratch_contents(host, encode_bf16(final_values), kTail);
    CHECK_EQ(device_scratch.owner_identity(), workspace_identity);
}

TEST_CASE("greedy token selection does not treat opaque logits as a host range") {
    struct Descriptor {
        int tag = 7;
        std::array<std::byte, 16> host;
    } descriptor;
    class OpaqueLogits final : public iom::Tensor {
    public:
        OpaqueLogits(iom::Device& device, Descriptor& descriptor, void* key)
                : Tensor(logits_spec(3), device), descriptor_(descriptor),
                  key_(key), payload_(encode_bf16(
                          std::vector<std::uint16_t>{0xbf80, 0x4000, 0x3f80})) {}
    private:
        iom::WorkspaceRequirements host_transfer_workspace_requirements(
                std::size_t) const override { return {0, 1}; }
        void* storage_handle() noexcept override { return &descriptor_; }
        iom::detail::StorageIdentity storage_identity() const noexcept override {
            return {key_, nullptr};
        }
        void region_from_host(const iom::TensorView&,
                std::span<const std::byte>, iom::RawWorkspaceView) override {
            throw std::logic_error("read-only logits fixture");
        }
        void region_to_host(const iom::TensorView&,
                std::span<std::byte> destination, iom::RawWorkspaceView) const override {
            std::copy(payload_.begin(), payload_.end(), destination.begin());
        }
        Descriptor& descriptor_;
        void* key_;
        std::vector<std::byte> payload_;
    };
    CpuFixture fixture;
    char key;
    OpaqueLogits logits(*fixture.device, descriptor, &key);
    ManualQueue queue(*fixture.device);
    const auto producer = queue.probe();
    queue.complete(ManualQueue::sequence(producer));
    iom::GreedyTokenSelector selector;
    descriptor.host.fill(std::byte{0x5a});
    const auto before = descriptor.host;
    CHECK_THROWS_AS(selector.select(
            queue, logits.view(), 3, iom::to_oid(iom::OidError::Unsupported),
            {}, {descriptor.host, {}}), std::invalid_argument);
    CHECK(descriptor.host == before);
    CHECK_THROWS_AS(selector.select(
            queue, logits.view(), 3, producer, {},
            {std::span<std::byte>(descriptor.host).first(5), {}}),
            std::invalid_argument);
    CHECK(descriptor.host == before);
    // Host scratch lies immediately after the real execution descriptor,
    // inside its would-be tiled byte range, but not inside opaque backing.
    CHECK_EQ(selector.select(
            queue, logits.view(), 3, producer, {}, {descriptor.host, {}}), 1);
    check_scratch_contents(descriptor.host,
            encode_bf16(std::vector<std::uint16_t>{0xbf80, 0x4000, 0x3f80}),
            std::byte{0x5a});
}
