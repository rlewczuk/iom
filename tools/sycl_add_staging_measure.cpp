// Disposable SYCL ADD measurement harness. It is intentionally outside the
// production build graph; compile it against a BUILD_TESTING SYCL build.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/iom.hpp"
#include "iom/sycl/device.hpp"
#include "runtime.hpp"

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kWarmup = 10;
constexpr std::size_t kDefaultIterations = 100;
constexpr std::size_t kBudget = 1ULL << 30;

struct Allocator final : iom::Allocator {
    void bind(const sycl::context& context) {
        context_ = context;
        devices_ = context.get_devices();
    }
    void* alloc(std::size_t bytes) override {
        if (!context_ || devices_.empty()) throw std::bad_alloc();
        void* result = sycl::malloc_shared(bytes, devices_.front(), *context_);
        if (result == nullptr) throw std::bad_alloc();
        return result;
    }
    void free(void* pointer) override {
        if (pointer != nullptr && context_) sycl::free(pointer, *context_);
    }
    void reset() override {}
    std::optional<sycl::context> context_;
    std::vector<sycl::device> devices_;
};

Allocator* active_allocator = nullptr;
void capture_context(const sycl::context& context) {
    if (active_allocator == nullptr) throw std::logic_error("allocator callback is not installed");
    active_allocator->bind(context);
}

struct ContextRestore {
    iom::sycl_detail::ContextCalls saved;
    ~ContextRestore() { iom::sycl_detail::context_calls = saved; }
};

double seconds(Clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

double percentile(const std::vector<double>& sorted, double p) {
    return sorted[static_cast<std::size_t>(p * (sorted.size() - 1))];
}

std::size_t bits(iom::DataType type) {
    return type == iom::DataType::BF16 ? 16 : type == iom::DataType::F32 ? 32 : 64;
}

std::size_t plane_bytes(const iom::TensorSpec& spec) {
    const auto dims = spec.shape.dimensions();
    const std::size_t rows = (dims[dims.size() - 2] + 15) / 16 * 16;
    const std::size_t columns = (dims.back() + 15) / 16 * 16;
    return rows * columns * bits(spec.data_type) / 8;
}

std::vector<std::byte> input(const iom::TensorSpec& spec, unsigned seed) {
    std::vector<std::byte> result(spec.logical_nbytes());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<std::byte>((seed + i * 17U) & 0xffU);
    return result;
}

struct AllocationTimes { double alloc_s = 0.0; double free_s = 0.0; };

AllocationTimes measure_allocations(
        const sycl::context& context, const sycl::device& device,
        std::size_t bytes, std::size_t iterations) {
    AllocationTimes result;
    for (std::size_t i = 0; i < iterations; ++i) {
        void* host[3]{};
        void* device_memory[3]{};
        const auto alloc_start = Clock::now();
        for (void*& pointer : host) pointer = sycl::malloc_host(bytes, context);
        for (void*& pointer : device_memory)
            pointer = sycl::malloc_device(bytes, device, context);
        result.alloc_s += seconds(Clock::now() - alloc_start);
        for (void* pointer : host) if (pointer == nullptr) throw std::bad_alloc();
        for (void* pointer : device_memory) if (pointer == nullptr) throw std::bad_alloc();
        const auto free_start = Clock::now();
        for (void* pointer : host) sycl::free(pointer, context);
        for (void* pointer : device_memory) sycl::free(pointer, context);
        result.free_s += seconds(Clock::now() - free_start);
    }
    return result;
}

double measure_host_loop_proxy(
        std::span<const std::byte> lhs, std::span<const std::byte> rhs,
        std::span<std::byte> out, std::size_t iterations) {
    const auto start = Clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = static_cast<std::byte>(
                    static_cast<unsigned char>(lhs[i]) +
                    static_cast<unsigned char>(rhs[i]));
    }
    return seconds(Clock::now() - start);
}

struct Result {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t logical_bytes = 0;
    std::size_t staging_bytes = 0;
    std::size_t iterations = 0;
    double min_s = 0.0;
    double median_s = 0.0;
    double p95_s = 0.0;
    double max_s = 0.0;
    double total_s = 0.0;
    double baseline_s = 0.0;
    double overlap_s = 0.0;
    AllocationTimes allocations;
    double host_loop_proxy_s = 0.0;
};

std::string dtype_name(iom::DataType type) {
    switch (type) {
    case iom::DataType::BF16: return "BF16";
    case iom::DataType::F32: return "F32";
    case iom::DataType::I64: return "I64";
    default: return "unknown";
    }
}

Result measure(
        iom::Device& device, iom::DeviceOps& ops, sycl::queue& independent,
        const sycl::context& context, const sycl::device& native_device,
        iom::DataType type, std::vector<std::size_t> shape,
        std::size_t iterations) {
    const iom::TensorSpec spec{iom::TensorShape(shape), type};
    const auto lhs_data = input(spec, 3);
    const auto rhs_data = input(spec, 7);
    std::vector<std::byte> proxy_out(lhs_data.size());
    auto lhs = device.create_tensor(spec);
    auto rhs = device.create_tensor(spec);
    auto out = device.create_tensor(spec);
    lhs->view().copy_from_host(lhs_data);
    rhs->view().copy_from_host(rhs_data);
    for (std::size_t i = 0; i < kWarmup; ++i) {
        const auto token = ops.add(lhs->view(), rhs->view(), out->view());
        if (!iom::oid_is_token(token)) throw std::runtime_error("SYCL ADD warm-up rejected");
        ops.wait(token);
    }

    std::vector<double> samples;
    std::vector<double> baseline;
    std::vector<double> overlap;
    samples.reserve(iterations);
    baseline.reserve(iterations);
    overlap.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto baseline_start = Clock::now();
        auto baseline_event = independent.parallel_for(
                sycl::range<1>(1024), [=](sycl::id<1>) {});
        baseline_event.wait_and_throw();
        baseline.push_back(seconds(Clock::now() - baseline_start));

        const auto independent_start = Clock::now();
        auto independent_event = independent.parallel_for(
                sycl::range<1>(1024), [=](sycl::id<1>) {});
        const auto add_start = Clock::now();
        const auto token = ops.add(lhs->view(), rhs->view(), out->view());
        if (!iom::oid_is_token(token)) throw std::runtime_error("SYCL ADD rejected");
        ops.wait(token);
        const auto add_end = Clock::now();
        independent_event.wait_and_throw();
        const auto independent_end = Clock::now();
        samples.push_back(seconds(add_end - add_start));
        const auto left = std::max(add_start, independent_start);
        const auto right = std::min(add_end, independent_end);
        overlap.push_back(right > left ? seconds(right - left) : 0.0);
    }
    std::sort(samples.begin(), samples.end());
    std::sort(baseline.begin(), baseline.end());
    std::sort(overlap.begin(), overlap.end());
    const AllocationTimes allocation_times = measure_allocations(
            context, native_device, plane_bytes(spec), iterations);
    const double host_loop = measure_host_loop_proxy(
            lhs_data, rhs_data, proxy_out, iterations);
    Result result;
    result.dtype = dtype_name(type);
    result.shape = std::move(shape);
    result.logical_bytes = spec.logical_nbytes();
    result.staging_bytes = plane_bytes(spec) *
                           (spec.shape.element_count() /
                            (result.shape[result.shape.size() - 2] *
                             result.shape.back()));
    result.iterations = iterations;
    result.min_s = samples.front();
    result.median_s = percentile(samples, .50);
    result.p95_s = percentile(samples, .95);
    result.max_s = samples.back();
    result.total_s = std::accumulate(samples.begin(), samples.end(), 0.0);
    result.baseline_s = percentile(baseline, .50);
    result.overlap_s = percentile(overlap, .50);
    result.allocations = allocation_times;
    result.host_loop_proxy_s = host_loop;
    return result;
}

void print_result(const Result& result) {
    const std::size_t planes = result.staging_bytes == 0 ? 0 :
            result.logical_bytes / (result.shape[result.shape.size() - 2] * result.shape.back() *
                                    (result.dtype == "BF16" ? 2 : result.dtype == "F32" ? 4 : 8));
    const std::size_t memcpy_bytes = 4 * result.staging_bytes;
    std::cout << std::setprecision(9)
              << "workload=" << result.dtype << " shape={";
    for (std::size_t i = 0; i < result.shape.size(); ++i)
        std::cout << (i ? "," : "") << result.shape[i];
    std::cout << "} logical_bytes=" << result.logical_bytes
              << " staging_bytes_per_buffer=" << result.staging_bytes
              << " native_planes=" << planes
              << " warmup=" << kWarmup << " iterations=" << result.iterations
              << " min_s=" << result.min_s << " median_s=" << result.median_s
              << " p95_s=" << result.p95_s << " max_s=" << result.max_s
              << " total_s=" << result.total_s
              << " throughput_Bps=" << (result.logical_bytes / result.median_s)
              << " usm_alloc_count=" << 6 * result.iterations
              << " usm_alloc_time_s=" << result.allocations.alloc_s
              << " usm_free_count=" << 6 * result.iterations
              << " usm_free_time_s=" << result.allocations.free_s
              << " kernel_launch_count=" << 4 * result.iterations
              << " memcpy_count=" << 4 * result.iterations
              << " memcpy_bytes=" << memcpy_bytes * result.iterations
              << " wait_and_throw_count=" << 2 * result.iterations
              << " wait_and_throw_time_s=" << result.total_s
              << " host_loop_proxy_time_s=" << result.host_loop_proxy_s
              << " staging_peak_bytes=" << 6 * result.staging_bytes
              << " independent_baseline_median_s=" << result.baseline_s
              << " overlap_median_s=" << result.overlap_s << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    std::size_t iterations = kDefaultIterations;
    if (argc == 3 && std::string_view(argv[1]) == "--iterations")
        iterations = std::max<std::size_t>(100, std::strtoull(argv[2], nullptr, 10));
    if (argc != 1 && argc != 3) return 2;
    ContextRestore restore;
    Allocator allocator;
    active_allocator = &allocator;
    iom::sycl_detail::context_calls.context_ready = &capture_context;
    const sycl::device selected(sycl::gpu_selector_v);
    if (selected.get_backend() != sycl::backend::ext_oneapi_level_zero)
        throw std::runtime_error("configured device is not Level Zero");
    std::cout << "declared_memory_budget_bytes=" << kBudget
              << " backend=LevelZero device="
              << selected.get_info<sycl::info::device::name>()
              << " driver=" << selected.get_info<sycl::info::device::driver_version>()
              << " compiler=" << __VERSION__ << " iterations_minimum=100\n";
    auto device = iom::make_sycl_device(0, allocator);
    auto ops = device->create_ops();
    sycl::queue independent(selected);
    for (const auto& [type, shape] : std::vector<std::pair<iom::DataType, std::vector<std::size_t>>>{
             {iom::DataType::BF16, {8, 1024, 1024}},
             {iom::DataType::F32, {8, 1024, 1024}},
             {iom::DataType::BF16, {1, 1024, 1024}},
             {iom::DataType::F32, {1, 1024, 1024}},
             {iom::DataType::I64, {2, 33, 65}}}) {
        print_result(measure(*device, *ops, independent, allocator.context_.value(),
                             selected, type, shape, iterations));
    }
}
