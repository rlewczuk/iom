#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/iom.hpp"
#include "iom/cpu/device.hpp"
#include "iom/tensor.hpp"

namespace {

class HostAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        return ::operator new(size, std::align_val_t(32));
    }

    void free(void* buffer) override {
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {}
};

template <typename Operation>
double median_seconds(Operation&& operation) {
    std::vector<double> samples;
    samples.reserve(5);
    operation();
    for (std::size_t run = 0; run < 5; ++run) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(
                std::chrono::duration<double>(finish - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct BenchmarkResult {
    double from_gigabytes_per_second;
    double to_gigabytes_per_second;
    double queued_gigabytes_per_second;
    double queued_small_seconds;
};

BenchmarkResult measure(
        const iom::TensorSpec& spec, bool measure_small_latency) {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::size_t bytes = spec.logical_nbytes();
    std::vector<std::byte> host_source(bytes, std::byte{0x5A});
    std::vector<std::byte> host_destination(bytes, std::byte{0});

    const double from_seconds = median_seconds([&] {
        source->view().copy_from_host(host_source);
    });
    const double to_seconds = median_seconds([&] {
        source->view().copy_to_host(host_destination);
    });

    auto queue = device->create_ops();
    const double queued_seconds = median_seconds([&] {
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        queue->wait(token);
    });

    return {
            static_cast<double>(bytes) / from_seconds / 1.0e9,
            static_cast<double>(bytes) / to_seconds / 1.0e9,
            static_cast<double>(bytes) / queued_seconds / 1.0e9,
            measure_small_latency ? queued_seconds : 0.0};
}

}  // namespace

TEST_CASE("CPU benchmark: blocked copy throughput vs memory floor") {
    const BenchmarkResult f32_large = measure(
            iom::TensorSpec{
                    iom::TensorShape{{4096, 4096}}, iom::DataType::F32},
            false);
    const BenchmarkResult i4_large = measure(
            iom::TensorSpec{
                    iom::TensorShape{{2048, 2048}}, iom::DataType::I4},
            false);
    const BenchmarkResult f32_small = measure(
            iom::TensorSpec{
                    iom::TensorShape{{16, 16}}, iom::DataType::F32},
            true);

    MESSAGE("F32 4096x4096 copy_from_host: "
            << f32_large.from_gigabytes_per_second << " GB/s");
    MESSAGE("F32 4096x4096 copy_to_host: "
            << f32_large.to_gigabytes_per_second << " GB/s");
    MESSAGE("F32 4096x4096 queued copy: "
            << f32_large.queued_gigabytes_per_second << " GB/s");
    MESSAGE("I4 2048x2048 copy_from_host: "
            << i4_large.from_gigabytes_per_second << " GB/s");
    MESSAGE("F32 16x16 queued submit+wait: "
            << f32_small.queued_small_seconds * 1.0e6 << " us");

    CHECK_GE(f32_large.from_gigabytes_per_second, 10.0);
    CHECK_GE(f32_large.to_gigabytes_per_second, 10.0);
    CHECK_GE(f32_large.queued_gigabytes_per_second, 5.0);
    CHECK_GE(i4_large.from_gigabytes_per_second, 1.0);
    CHECK_LE(f32_small.queued_small_seconds, 6.0e-6);
}
