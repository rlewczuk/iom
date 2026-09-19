#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
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
