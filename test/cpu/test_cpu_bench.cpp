#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <new>
#include <sstream>
#include <string>
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

std::string format_microseconds(double seconds) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << seconds * 1.0e6;
    return stream.str();
}

struct BenchmarkResult {
    double from_gigabytes_per_second;
    double to_gigabytes_per_second;
    double queued_gigabytes_per_second;
    double queued_small_seconds;
    double queued_no_op_seconds;
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
    const auto measure_copy = [&] {
        const auto start = std::chrono::steady_clock::now();
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        queue->wait(token);
        const auto finish = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finish - start).count();
    };

    double queued_seconds = 0.0;
    double queued_no_op_seconds = 0.0;
    if (measure_small_latency) {
        const auto measure_no_op = [&] {
            const auto start = std::chrono::steady_clock::now();
            const iom::oid token =
                    queue->copy(source->view(), source->view());
            queue->wait(token);
            const auto finish = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(finish - start).count();
        };

        std::vector<double> copy_samples;
        std::vector<double> no_op_samples;
        copy_samples.reserve(11);
        no_op_samples.reserve(11);

        measure_no_op();
        measure_copy();
        for (std::size_t run = 0; run < 11; ++run) {
            no_op_samples.push_back(measure_no_op());
            copy_samples.push_back(measure_copy());
        }

        std::sort(copy_samples.begin(), copy_samples.end());
        std::sort(no_op_samples.begin(), no_op_samples.end());
        queued_seconds = copy_samples[5];
        queued_no_op_seconds = no_op_samples[5];
    } else {
        queued_seconds = median_seconds(measure_copy);
    }

    return {
            static_cast<double>(bytes) / from_seconds / 1.0e9,
            static_cast<double>(bytes) / to_seconds / 1.0e9,
            static_cast<double>(bytes) / queued_seconds / 1.0e9,
            measure_small_latency ? queued_seconds : 0.0,
            measure_small_latency ? queued_no_op_seconds : 0.0};
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
    MESSAGE("F32 16x16 queued submit+wait (copy): "
            << f32_small.queued_small_seconds * 1.0e6 << " us");
    MESSAGE("F32 16x16 queued submit+wait (no-op): "
            << f32_small.queued_no_op_seconds * 1.0e6 << " us");
    MESSAGE("F32 16x16 queued submit+wait differential: "
            << (f32_small.queued_small_seconds -
                       f32_small.queued_no_op_seconds) *
                       1.0e6
            << " us");
    MESSAGE(
            "CPU benchmark reference host: AMD Ryzen AI 9 HX 370, Linux "
            "x86-64, g++ 15.2.0, Release -O2, single-threaded, ordinary "
            "developer load");

    // Queued-latency calibration on 2026-09-06, reference host:
    // AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release -O2,
    // single-threaded, ordinary developer load.
    // N = 20; R = 1e-9 seconds
    // (steady_clock::period::num / steady_clock::period::den).
    // D = -1.245e-7; Q = 2.655e-7; diff_max = 3.91e-7.
    // M = max(2 * Q, 10 * R) = 5.31e-7 seconds.
    // A_seconds = max(0, diff_max) + M + 2 * Q = 1.453e-6.
    // G = 1.0e-8 seconds; raw diff_median_1..20 (seconds):
    // {-9.9e-8, 1.6e-7, -4.1e-8, -9.72e-7, -4.4e-7,
    //  -3.01e-7, 1.4e-7, -1.5e-7, 1.71e-7, 3.91e-7,
    //  -9.41e-7, -3.91e-7, 1.51e-7, -3.41e-7, -9.21e-7,
    //  -3.11e-7, 3.1e-7, 5.0e-8, -9.0e-8, -2.01e-7}.
    // Injector = max(100e-6, 10 * kAllowanceSeconds) = 100e-6 seconds.
    // Rounded: kAllowanceSeconds = ceil(A_seconds / G) * G = 1.46e-6.
    constexpr double kAllowanceSeconds = 1.46e-6;

    // Post-CC-001 throughput calibration on 2026-09-06, reference host:
    // AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`,
    // single-threaded, ordinary developer load. N = 20 medians:
    // F32 copy_from_host = 8.046555 GB/s; copy_to_host = 4.890885 GB/s;
    // queued copy = 4.126280 GB/s; I4 copy_from_host = 1.322525 GB/s.
    // Floors are half each median, rounded down to 0.1 GB/s.
    CHECK_GE(f32_large.from_gigabytes_per_second, 4.0);
    CHECK_GE(f32_large.to_gigabytes_per_second, 2.4);
    CHECK_GE(f32_large.queued_gigabytes_per_second, 2.0);
    CHECK_GE(i4_large.from_gigabytes_per_second, 0.6);
    CHECK_MESSAGE(
            f32_small.queued_small_seconds <=
                    f32_small.queued_no_op_seconds + kAllowanceSeconds,
            "queued copy median = "
                    << format_microseconds(
                               f32_small.queued_small_seconds)
                    << " us, no-op median = "
                    << format_microseconds(
                               f32_small.queued_no_op_seconds)
                    << " us, differential = "
                    << format_microseconds(
                               f32_small.queued_small_seconds -
                               f32_small.queued_no_op_seconds)
                    << " us, allowance = "
                    << format_microseconds(kAllowanceSeconds) << " us");
}
