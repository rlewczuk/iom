// CPU copy throughput/latency benchmark.
//
// Generic CTest runs are reporting-only: the same workload as always is
// measured and every measurement is printed (workload, compiler/OS, detected
// hardware, warmup/sample counts, and sample distributions), but absolute
// throughput/latency floors never fail a generic run — one host's
// calibration must not gate correctness suites on slower or differently
// loaded machines.
//
// Threshold enforcement is a deliberate controlled-run contract: the binary
// asserts the calibrated floors and latency allowance only when
// IOM_CPU_BENCH_ENFORCE_FLOORS=1 is set AND the detected runner matches the
// calibrated reference host (AMD Ryzen AI 9 HX 370, Linux x86-64). Functional
// CPU correctness stays with the separate iom_cpu_tests and
// iom_backend_conformance_cpu_tests suites, which are unchanged.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/iom.hpp"
#include "iom/cpu/device.hpp"
#include "iom/tensor.hpp"

namespace {

constexpr std::size_t kThroughputWarmupRuns = 1;
constexpr std::size_t kThroughputSampleRuns = 5;
constexpr std::size_t kLatencyWarmupRuns = 1;
constexpr std::size_t kLatencySampleRuns = 11;

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

// One un-timed warmup run followed by `sample_count` timed runs; returns the
// sorted per-run durations in seconds. This is the original benchmark's
// timing methodology (one warmup, five timed samples, median of the sorted
// samples) extended to retain every sample so min/max/variance stay
// reportable alongside the median.
template <typename Operation>
std::vector<double> timed_samples(
        std::size_t sample_count, Operation&& operation) {
    std::vector<double> samples;
    samples.reserve(sample_count);
    operation();
    for (std::size_t run = 0; run < sample_count; ++run) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(
                std::chrono::duration<double>(finish - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

struct SampleStats {
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double standard_deviation = 0.0;
};

// Median/min/max and population standard deviation of sorted samples; the
// median is the middle timed sample, as in the original benchmark.
SampleStats sample_stats(const std::vector<double>& sorted_samples) {
    SampleStats stats;
    if (sorted_samples.empty()) {
        return stats;
    }
    const double mean =
            std::accumulate(sorted_samples.begin(), sorted_samples.end(), 0.0) /
            static_cast<double>(sorted_samples.size());
    double squared_deviation_sum = 0.0;
    for (double sample : sorted_samples) {
        const double deviation = sample - mean;
        squared_deviation_sum += deviation * deviation;
    }
    stats.median = sorted_samples[sorted_samples.size() / 2];
    stats.min = sorted_samples.front();
    stats.max = sorted_samples.back();
    stats.standard_deviation =
            std::sqrt(squared_deviation_sum / sorted_samples.size());
    return stats;
}

std::string format_microseconds(double seconds) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1) << seconds * 1.0e6;
    return stream.str();
}

struct ThroughputStats {
    double gigabytes_per_second = 0.0;
    double min_gigabytes_per_second = 0.0;
    double max_gigabytes_per_second = 0.0;
    double standard_deviation_gigabytes_per_second = 0.0;
};

// Sorted per-run durations -> GB/s sample distribution for `bytes` copied.
// The reciprocal conversion is strictly monotone, so the median GB/s equals
// the original factor / median-seconds metric.
ThroughputStats throughput_stats(
        const std::vector<double>& sorted_seconds, std::size_t bytes) {
    const double factor = static_cast<double>(bytes) / 1.0e9;
    std::vector<double> samples_in_gbps;
    samples_in_gbps.reserve(sorted_seconds.size());
    for (double seconds : sorted_seconds) {
        samples_in_gbps.push_back(factor / seconds);
    }
    std::sort(samples_in_gbps.begin(), samples_in_gbps.end());
    const SampleStats stats = sample_stats(samples_in_gbps);
    return {stats.median, stats.min, stats.max, stats.standard_deviation};
}

struct BenchmarkResult {
    ThroughputStats from;
    ThroughputStats to;
    ThroughputStats queued;
    SampleStats copy_latency_seconds;
    SampleStats no_op_latency_seconds;
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

    const std::vector<double> from_samples = timed_samples(
            kThroughputSampleRuns,
            [&] { source->view().copy_from_host(host_source); });
    const std::vector<double> to_samples = timed_samples(
            kThroughputSampleRuns,
            [&] { source->view().copy_to_host(host_destination); });

    auto queue = device->create_ops();
    const auto measure_copy = [&] {
        const auto start = std::chrono::steady_clock::now();
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        queue->wait(token);
        const auto finish = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finish - start).count();
    };

    std::vector<double> queued_samples;
    std::vector<double> no_op_samples;
    if (measure_small_latency) {
        const auto measure_no_op = [&] {
            const auto start = std::chrono::steady_clock::now();
            const iom::oid token =
                    queue->copy(source->view(), source->view());
            queue->wait(token);
            const auto finish = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(finish - start).count();
        };

        measure_no_op();
        measure_copy();
        queued_samples.reserve(kLatencySampleRuns);
        no_op_samples.reserve(kLatencySampleRuns);
        for (std::size_t run = 0; run < kLatencySampleRuns; ++run) {
            no_op_samples.push_back(measure_no_op());
            queued_samples.push_back(measure_copy());
        }
        std::sort(queued_samples.begin(), queued_samples.end());
        std::sort(no_op_samples.begin(), no_op_samples.end());
    } else {
        queued_samples = timed_samples(kThroughputSampleRuns, measure_copy);
    }

    return {
            throughput_stats(from_samples, bytes),
            throughput_stats(to_samples, bytes),
            throughput_stats(queued_samples, bytes),
            sample_stats(queued_samples),
            sample_stats(no_op_samples)};
}

// Detected hardware identity. Linux exposes the CPU model through
// /proc/cpuinfo; other hosts report it as unavailable.
#if defined(__linux__)
std::string detected_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (!line.starts_with("model name")) {
            continue;
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::size_t first_non_space =
                line.find_first_not_of(" \t", colon + 1);
        if (first_non_space == std::string::npos) {
            return "<unavailable>";
        }
        return line.substr(first_non_space);
    }
    return "<unavailable>";
}
#else
std::string detected_cpu_model() { return "<unavailable>"; }
#endif

// Compiler/OS/architecture/build identity, reported alongside the samples so
// a measurement can be attributed to the environment that produced it.
std::string compiler_os_identity() {
    std::ostringstream stream;
#if defined(__clang__)
    stream << "clang " << __clang_major__ << '.' << __clang_minor__ << '.'
           << __clang_patchlevel__;
#elif defined(__GNUC__)
    stream << "g++ " << __GNUC__ << '.' << __GNUC_MINOR__ << '.'
           << __GNUC_PATCHLEVEL__;
#else
    stream << "<unknown compiler>";
#endif
    stream << " | ";
#if defined(__linux__)
    stream << "Linux";
#elif defined(__APPLE__)
    stream << "macOS";
#elif defined(_WIN32)
    stream << "Windows";
#else
    stream << "<unknown OS>";
#endif
    stream << " | ";
#if defined(__x86_64__) || defined(_M_X64)
    stream << "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    stream << "AArch64";
#else
    stream << "<unknown architecture>";
#endif
    stream << " | ";
#if defined(NDEBUG)
    stream << "release build";
#else
    stream << "non-release build";
#endif
    return stream.str();
}

// Controlled-run gate: the calibrated floors are enforced only when the
// operator explicitly opts in to a controlled benchmark run on the reference
// host. No generic CTest invocation sets this variable, so generic runs are
// always reporting-only.
bool floors_enforcement_requested() {
    const char* raw = std::getenv("IOM_CPU_BENCH_ENFORCE_FLOORS");
    if (raw == nullptr) {
        return false;
    }
    const std::string value(raw);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "yes" || value == "YES";
}

// The runner the floors were calibrated on (see calibration comments below).
const char kReferenceCpuModel[] = "AMD Ryzen AI 9 HX 370";

bool runner_matches_reference() {
#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
    return detected_cpu_model().find(kReferenceCpuModel) != std::string::npos;
#else
    return false;
#endif
}

}  // namespace

TEST_CASE("CPU benchmark: blocked copy throughput vs memory floor") {
    // Workload unchanged: 4096x4096 F32 and 2048x2048 I4 host/queued
    // transfers plus the F32 16x16 queued submit+wait latency pair.
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
            << f32_large.from.gigabytes_per_second << " GB/s");
    MESSAGE("F32 4096x4096 copy_to_host: "
            << f32_large.to.gigabytes_per_second << " GB/s");
    MESSAGE("F32 4096x4096 queued copy: "
            << f32_large.queued.gigabytes_per_second << " GB/s");
    MESSAGE("I4 2048x2048 copy_from_host: "
            << i4_large.from.gigabytes_per_second << " GB/s");
    MESSAGE("F32 16x16 queued submit+wait (copy): "
            << f32_small.copy_latency_seconds.median * 1.0e6 << " us");
    MESSAGE("F32 16x16 queued submit+wait (no-op): "
            << f32_small.no_op_latency_seconds.median * 1.0e6 << " us");
    MESSAGE("F32 16x16 queued submit+wait differential: "
            << (f32_small.copy_latency_seconds.median -
                       f32_small.no_op_latency_seconds.median) *
                       1.0e6
            << " us");
    MESSAGE(
            "CPU benchmark reference host: AMD Ryzen AI 9 HX 370, Linux "
            "x86-64, g++ 15.2.0, Release -O2, single-threaded, ordinary "
            "developer load");
    MESSAGE("Detected runner: CPU = \"" << detected_cpu_model()
            << "\", threads = " << std::thread::hardware_concurrency()
            << ", identity = " << compiler_os_identity());
    MESSAGE("Timing methodology: "
            << kThroughputWarmupRuns << " warmup + " << kThroughputSampleRuns
            << " timed samples per throughput/queued measurement, and "
            << kLatencyWarmupRuns << " warmup + " << kLatencySampleRuns
            << " for the F32 16x16 latency pair; medians of the sorted "
               "samples");
    MESSAGE("Sample distributions (min / median / max / sigma):");
    MESSAGE("  F32 copy_from_host: "
            << f32_large.from.min_gigabytes_per_second << " / "
            << f32_large.from.gigabytes_per_second << " / "
            << f32_large.from.max_gigabytes_per_second << " / "
            << f32_large.from.standard_deviation_gigabytes_per_second
            << " GB/s");
    MESSAGE("  F32 copy_to_host: "
            << f32_large.to.min_gigabytes_per_second << " / "
            << f32_large.to.gigabytes_per_second << " / "
            << f32_large.to.max_gigabytes_per_second << " / "
            << f32_large.to.standard_deviation_gigabytes_per_second
            << " GB/s");
    MESSAGE("  F32 queued copy: "
            << f32_large.queued.min_gigabytes_per_second << " / "
            << f32_large.queued.gigabytes_per_second << " / "
            << f32_large.queued.max_gigabytes_per_second << " / "
            << f32_large.queued.standard_deviation_gigabytes_per_second
            << " GB/s");
    MESSAGE("  I4 copy_from_host: "
            << i4_large.from.min_gigabytes_per_second << " / "
            << i4_large.from.gigabytes_per_second << " / "
            << i4_large.from.max_gigabytes_per_second << " / "
            << i4_large.from.standard_deviation_gigabytes_per_second
            << " GB/s");
    MESSAGE("  F32 16x16 queued submit+wait copy: "
            << format_microseconds(f32_small.copy_latency_seconds.min)
            << " / "
            << format_microseconds(f32_small.copy_latency_seconds.median)
            << " / "
            << format_microseconds(f32_small.copy_latency_seconds.max)
            << " / "
            << format_microseconds(
                       f32_small.copy_latency_seconds.standard_deviation)
            << " us");
    MESSAGE("  F32 16x16 queued submit+wait no-op: "
            << format_microseconds(f32_small.no_op_latency_seconds.min)
            << " / "
            << format_microseconds(f32_small.no_op_latency_seconds.median)
            << " / "
            << format_microseconds(f32_small.no_op_latency_seconds.max)
            << " / "
            << format_microseconds(
                       f32_small.no_op_latency_seconds.standard_deviation)
            << " us");

    // Queued-latency calibration on 2026-09-06, reference host:
    // AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`,
    // single-threaded, ordinary developer load.
    // N = 20; R = 1e-9 seconds
    // (steady_clock::period::num / steady_clock::period::den).
    // D = 6.0e-8; Q = 0.0; diff_max = 6.0e-8 seconds.
    // M = max(2 * Q, 10 * R) = 1.0e-8 seconds.
    // A_seconds = max(0, diff_max) + M + 2 * Q = 7.0e-8 seconds.
    // G = 1.0e-8 seconds; raw diff_median_1..20 (seconds):
    // {6.0e-8, 5.1e-8, 6.0e-8, 6.0e-8, 5.0e-8,
    //  6.0e-8, 5.1e-8, 5.1e-8, 6.0e-8, 6.0e-8,
    //  6.0e-8, 6.0e-8, 6.0e-8, 5.1e-8, 5.9e-8,
    //  6.0e-8, 6.0e-8, 5.0e-8, 5.9e-8, 6.0e-8}.
    // Injector = max(100, duration_cast<microseconds>(
    //     10 * kAllowanceSeconds).count()) = 100 us.
    // Rounded: kAllowanceSeconds = ceil(A_seconds / G) * G = 7.0e-8.
    // This post-73 re-derivation supersedes order 67's constant because
    // order 73 deleted the CPU StagedWorker path (CpuQueue::Task, execute,
    // complete_task, worker_, and shutdown_and_drain).
    constexpr double kAllowanceSeconds = 7.0e-8;

    // Post-CC-001 throughput calibration on 2026-09-06, reference host:
    // AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`,
    // single-threaded, ordinary developer load. N = 20 medians:
    // F32 copy_from_host = 8.046555 GB/s; copy_to_host = 4.890885 GB/s;
    // queued copy = 4.126280 GB/s; I4 copy_from_host = 1.322525 GB/s.
    // Floors are half each median, rounded down to 0.1 GB/s. They gate only
    // the controlled-run contract below; generic CTest runs never assert
    // them.
    constexpr double kF32FromHostFloorGbps = 4.0;
    constexpr double kF32ToHostFloorGbps = 2.4;
    constexpr double kF32QueuedFloorGbps = 2.0;
    constexpr double kI4FromHostFloorGbps = 0.6;

    // Thresholds are a controlled-run contract: explicit opt-in through the
    // IOM_CPU_BENCH_ENFORCE_FLOORS environment variable AND a runner that
    // matches the calibrated reference host. Generic CTest runs (no
    // variable) only report measurements; a mismatched runner that opts in
    // is reported and left unasserted rather than failing on floors it was
    // never calibrated for.
    if (!floors_enforcement_requested()) {
        MESSAGE(
                "Generic run: neither the throughput floors nor the queued-"
                "latency allowance are enforced (reporting only). To gate "
                "on the calibrated floors, run a controlled benchmark job "
                "on the reference host with "
                "IOM_CPU_BENCH_ENFORCE_FLOORS=1 set. Functional CPU "
                "correctness is covered by iom_cpu_tests and "
                "iom_backend_conformance_cpu_tests.");
        return;
    }

    MESSAGE("Controlled run: threshold enforcement requested"
            << " (IOM_CPU_BENCH_ENFORCE_FLOORS=1)");
    if (!runner_matches_reference()) {
        MESSAGE("Controlled-run contract not satisfied: detected runner is "
                << "not the calibrated reference host (\"" << kReferenceCpuModel
                << "\", Linux x86-64); thresholds NOT enforced, results "
                   "reported for comparison only.");
        return;
    }

    MESSAGE("Controlled-run contract satisfied: runner matches the "
            << "reference host; enforcing floors calibrated at half the "
               "reference medians (AMD Ryzen AI 9 HX 370, Linux x86-64, "
               "g++ 15.2.0, Release -O2, single-threaded, ordinary "
               "developer load)");
    MESSAGE("Threshold decision -- F32 copy_from_host: "
            << f32_large.from.gigabytes_per_second << " GB/s vs floor "
            << kF32FromHostFloorGbps << " GB/s");
    CHECK_GE(f32_large.from.gigabytes_per_second, kF32FromHostFloorGbps);
    MESSAGE("Threshold decision -- F32 copy_to_host: "
            << f32_large.to.gigabytes_per_second << " GB/s vs floor "
            << kF32ToHostFloorGbps << " GB/s");
    CHECK_GE(f32_large.to.gigabytes_per_second, kF32ToHostFloorGbps);
    MESSAGE("Threshold decision -- F32 queued copy: "
            << f32_large.queued.gigabytes_per_second << " GB/s vs floor "
            << kF32QueuedFloorGbps << " GB/s");
    CHECK_GE(f32_large.queued.gigabytes_per_second, kF32QueuedFloorGbps);
    MESSAGE("Threshold decision -- I4 copy_from_host: "
            << i4_large.from.gigabytes_per_second << " GB/s vs floor "
            << kI4FromHostFloorGbps << " GB/s");
    CHECK_GE(i4_large.from.gigabytes_per_second, kI4FromHostFloorGbps);
    MESSAGE("Threshold decision -- queued latency differential: "
            << format_microseconds(f32_small.copy_latency_seconds.median -
                                   f32_small.no_op_latency_seconds.median)
            << " us vs allowance " << format_microseconds(kAllowanceSeconds)
            << " us");
    CHECK_MESSAGE(
            f32_small.copy_latency_seconds.median <=
                    f32_small.no_op_latency_seconds.median +
                            kAllowanceSeconds,
            "queued copy median = "
                    << format_microseconds(
                               f32_small.copy_latency_seconds.median)
                    << " us, no-op median = "
                    << format_microseconds(
                               f32_small.no_op_latency_seconds.median)
                    << " us, differential = "
                    << format_microseconds(
                               f32_small.copy_latency_seconds.median -
                               f32_small.no_op_latency_seconds.median)
                    << " us, allowance = "
                    << format_microseconds(kAllowanceSeconds) << " us");
}