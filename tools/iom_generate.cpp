#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/session.hpp"

#if defined(IOM_GENERATE_CUDA_ENABLED)
#include "iom/cuda/device.hpp"
#endif
#if defined(IOM_GENERATE_ROCM_ENABLED)
#include "iom/rocm/device.hpp"
#endif
#if defined(IOM_GENERATE_SYCL_ENABLED)
#include "iom/sycl/device.hpp"
#endif

namespace {

class UsageError final : public std::runtime_error {
public:
    explicit UsageError(std::string message)
        : std::runtime_error(std::move(message)) {}
};

class SetupError final : public std::runtime_error {
public:
    explicit SetupError(std::string message)
        : std::runtime_error(std::move(message)) {}
};

class ProcessAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t bytes) override {
        return ::operator new(
                bytes == 0 ? std::size_t{1} : bytes,
                std::align_val_t{32});
    }

    void free(void* buffer) override {
        if (buffer != nullptr) {
            ::operator delete(buffer, std::align_val_t{32});
        }
    }

    void reset() override {}
};

enum class Backend { cpu, cuda, rocm, sycl };

struct ParsedMessage {
    std::string role;
    std::string content;
};

struct Arguments {
    std::string model_directory;
    Backend backend = Backend::cpu;
    std::uint32_t device = 0;
    std::size_t max_new_tokens = 0;
    std::size_t tensor_arena_bytes = 0;
    bool has_model_directory = false;
    bool has_backend = false;
    bool has_device = false;
    bool has_max_new_tokens = false;
    bool has_tensor_arena_bytes = false;
    bool trace = false;
    std::optional<std::string> prompt;
    std::vector<ParsedMessage> messages;
};

[[noreturn]] void usage_error(std::string message) {
    throw UsageError(std::move(message));
}

[[nodiscard]] std::string_view next_value(
        int argc, char** argv, int& index, std::string_view option) {
    if (index + 1 >= argc) {
        usage_error("missing value for " + std::string(option));
    }
    ++index;
    return argv[index];
}

[[nodiscard]] std::size_t parse_size(
        std::string_view text, std::string_view option) {
    if (text.empty()) {
        usage_error("empty value for " + std::string(option));
    }

    std::size_t value = 0;
    const auto parsed = std::from_chars(
            text.data(), text.data() + text.size(), value, 10);
    if (parsed.ec == std::errc::result_out_of_range) {
        usage_error("value overflows size_t for " + std::string(option));
    }
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        usage_error("invalid nonnegative integer for " + std::string(option));
    }
    return value;
}

[[nodiscard]] Backend parse_backend(std::string_view text) {
    if (text == "cpu") return Backend::cpu;
    if (text == "cuda") return Backend::cuda;
    if (text == "rocm") return Backend::rocm;
    if (text == "sycl") return Backend::sycl;
    usage_error("backend must be one of cpu, cuda, rocm, or sycl");
}

[[nodiscard]] Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--model-dir") {
            if (arguments.has_model_directory) {
                usage_error("duplicate --model-dir");
            }
            arguments.model_directory = std::string(
                    next_value(argc, argv, index, option));
            arguments.has_model_directory = true;
        } else if (option == "--backend") {
            if (arguments.has_backend) {
                usage_error("duplicate --backend");
            }
            arguments.backend = parse_backend(
                    next_value(argc, argv, index, option));
            arguments.has_backend = true;
        } else if (option == "--device") {
            if (arguments.has_device) {
                usage_error("duplicate --device");
            }
            const std::size_t ordinal = parse_size(
                    next_value(argc, argv, index, option), option);
            if (ordinal > std::numeric_limits<std::uint32_t>::max()) {
                usage_error("device ordinal is outside the backend range");
            }
            arguments.device = static_cast<std::uint32_t>(ordinal);
            arguments.has_device = true;
        } else if (option == "--max-new-tokens") {
            if (arguments.has_max_new_tokens) {
                usage_error("duplicate --max-new-tokens");
            }
            arguments.max_new_tokens = parse_size(
                    next_value(argc, argv, index, option), option);
            arguments.has_max_new_tokens = true;
        } else if (option == "--tensor-arena-bytes") {
            if (arguments.has_tensor_arena_bytes) {
                usage_error("duplicate --tensor-arena-bytes");
            }
            arguments.tensor_arena_bytes = parse_size(
                    next_value(argc, argv, index, option), option);
            arguments.has_tensor_arena_bytes = true;
        } else if (option == "--trace") {
            if (arguments.trace) {
                usage_error("duplicate --trace");
            }
            arguments.trace = true;
        } else if (option == "--prompt") {
            if (arguments.prompt.has_value()) {
                usage_error("duplicate --prompt");
            }
            arguments.prompt = std::string(
                    next_value(argc, argv, index, option));
        } else if (option == "--message") {
            const std::string role(
                    next_value(argc, argv, index, option));
            const std::string content(
                    next_value(argc, argv, index, option));
            arguments.messages.push_back(
                    ParsedMessage{std::move(role), std::move(content)});
        } else {
            usage_error("unknown option or extra positional argument: "
                        + std::string(option));
        }
    }

    if (!arguments.has_model_directory || !arguments.has_backend
            || !arguments.has_device || !arguments.has_max_new_tokens) {
        usage_error(
                "--model-dir, --backend, --device, and --max-new-tokens "
                "are required");
    }
    if (arguments.backend == Backend::cpu) {
        if (arguments.device != 0) {
            usage_error("CPU supports only device ordinal 0");
        }
        if (arguments.has_tensor_arena_bytes) {
            usage_error("--tensor-arena-bytes is only valid for accelerators");
        }
    } else {
        if (!arguments.has_tensor_arena_bytes) {
            usage_error(
                    "accelerator backends require --tensor-arena-bytes");
        }
        if (arguments.tensor_arena_bytes == 0
                || arguments.tensor_arena_bytes % 32 != 0) {
            usage_error(
                    "--tensor-arena-bytes must be nonzero and divisible by 32");
        }
    }

    const bool has_prompt = arguments.prompt.has_value();
    const bool has_messages = !arguments.messages.empty();
    if (has_prompt == has_messages) {
        usage_error(
                "provide exactly one --prompt or one or more --message pairs");
    }
    return arguments;
}

[[nodiscard]] std::unique_ptr<iom::Device> make_device(
        const Arguments& arguments, ProcessAllocator& cpu_allocator) {
    switch (arguments.backend) {
    case Backend::cpu:
        return iom::make_cpu_device(cpu_allocator);
    case Backend::cuda:
#if defined(IOM_GENERATE_CUDA_ENABLED)
        return iom::make_cuda_device(
                arguments.device,
                iom::DeviceMemoryConfig{arguments.tensor_arena_bytes});
#else
        throw SetupError("CUDA backend is not built");
#endif
    case Backend::rocm:
#if defined(IOM_GENERATE_ROCM_ENABLED)
        return iom::make_rocm_device(
                arguments.device,
                iom::DeviceMemoryConfig{arguments.tensor_arena_bytes});
#else
        throw SetupError("ROCm backend is not built");
#endif
    case Backend::sycl:
#if defined(IOM_GENERATE_SYCL_ENABLED)
        return iom::make_sycl_device(
                arguments.device,
                iom::DeviceMemoryConfig{arguments.tensor_arena_bytes});
#else
        throw SetupError("SYCL backend is not built");
#endif
    }
    throw SetupError("unknown backend selection");
}

void report(std::string_view category, std::string_view cause) {
    std::cerr << "iom_generate: " << category << ": " << cause << '\n';
}

void report_exception(std::string_view category, const std::exception& error) {
    report(category, error.what());
}

[[nodiscard]] const char* trace_phase_name(
        iom::InferencePhase phase) noexcept {
    switch (phase) {
    case iom::InferencePhase::load:
        return "load";
    case iom::InferencePhase::tokenization:
        return "tokenization";
    case iom::InferencePhase::prefill:
        return "prefill";
    case iom::InferencePhase::decode:
        return "decode";
    }
    return "unknown";
}

[[nodiscard]] const char* trace_wait_state_name(
        iom::WaitState state) noexcept {
    switch (state) {
    case iom::WaitState::not_observed:
        return "not_observed";
    case iom::WaitState::succeeded:
        return "succeeded";
    case iom::WaitState::failed:
        return "failed";
    }
    return "unknown";
}

[[nodiscard]] std::uint64_t trace_elapsed(
        std::uint64_t begin, std::uint64_t end) noexcept {
    return end >= begin ? end - begin : 0;
}

// Operation traces are a secondary presentation.  A stream or formatting
// failure must never replace a primary input, load, or execution result.
void report_operation_trace(const iom::InferenceMetrics& recorder) noexcept {
    try {
        for (const iom::InferenceTraceRecord& operation :
             recorder.operations()) {
            std::cerr << "iom_generate: trace"
                      << " request_ordinal=" << operation.request_ordinal
                      << " oid=" << operation.operation
                      << " phase=" << trace_phase_name(operation.phase)
                      << " layer=";
            if (operation.decoder_layer.has_value()) {
                std::cerr << *operation.decoder_layer;
            } else {
                std::cerr << "none";
            }
            std::cerr << " position_start=" << operation.position_start
                      << " run_length=" << operation.run_length
                      << " host_enqueue elapsed_ns="
                      << trace_elapsed(operation.enqueue_begin_ns,
                                      operation.enqueue_end_ns);
            if (operation.wait_observed_ns.has_value()) {
                std::cerr << " completion_observed elapsed_ns="
                          << trace_elapsed(operation.enqueue_begin_ns,
                                          *operation.wait_observed_ns);
            }
            std::cerr << " wait state="
                      << trace_wait_state_name(operation.wait_state)
                      << " device_time=unavailable\n";
            if (!std::cerr) break;
        }
        std::cerr.flush();
    } catch (...) {
        // Trace reporting is deliberately best-effort and never changes the
        // status or diagnostic selected by the primary operation.
    }
}

[[nodiscard]] int execute(const Arguments& arguments) {
    // Declaration order is deliberate: the CPU allocator outlives the device,
    // the device outlives the session, and the session outlives the result.
    // The optional recorder is declared before the session so its storage and
    // clock remain alive through ordinary session destruction and its final
    // queue drain.
    ProcessAllocator cpu_allocator;
    std::unique_ptr<iom::InferenceMetrics> recorder;
    std::unique_ptr<iom::Device> device;
    std::unique_ptr<iom::TinyLlamaSession> session;
    try {
        if (arguments.trace) {
            recorder = std::make_unique<iom::InferenceMetrics>();
        }
        device = make_device(arguments, cpu_allocator);
        if (recorder) {
            session = iom::load_tinyllama_session(
                    std::filesystem::path(arguments.model_directory), *device,
                    std::make_unique<iom::GreedyTokenSelector>(),
                    recorder.get());
            // Trace storage is reserved only after load succeeds and before
            // the first generation request.  A failed reservation is setup
            // failure, never an untraced generation.
            if (arguments.trace) {
                session->prepare_operation_trace();
            }
        } else {
            session = iom::load_tinyllama_session(
                    std::filesystem::path(arguments.model_directory), *device);
        }
    } catch (const std::exception& error) {
        report_exception("setup/load", error);
        return 3;
    } catch (...) {
        report("setup/load", "unknown setup failure");
        return 3;
    }

    std::vector<iom::ChatMessageView> message_views;
    std::optional<iom::GenerationResult> result;
    int status = 0;
    try {
        if (arguments.prompt.has_value()) {
            result.emplace(session->generate_raw(
                    *arguments.prompt, arguments.max_new_tokens));
        } else {
            message_views.reserve(arguments.messages.size());
            for (const ParsedMessage& message : arguments.messages) {
                message_views.push_back(
                        iom::ChatMessageView{message.role, message.content});
            }
            result.emplace(session->generate_chat(
                    std::span<const iom::ChatMessageView>(message_views),
                    arguments.max_new_tokens));
        }
    } catch (const std::invalid_argument& error) {
        report_exception("usage/input", error);
        status = 2;
    } catch (const std::exception& error) {
        report_exception("execution", error);
        status = 4;
    } catch (...) {
        report("execution", "unknown generation failure");
        status = 4;
    }

    if (status == 0) {
        try {
            std::cout.write(result->text.data(),
                            static_cast<std::streamsize>(result->text.size()));
            std::cout.flush();
            if (!std::cout) {
                throw std::runtime_error("decoded output write failed");
            }
        } catch (const std::exception& error) {
            report_exception("execution", error);
            status = 4;
        } catch (...) {
            report("execution", "unknown output failure");
            status = 4;
        }
    }

    // Release the session explicitly before reporting.  Its normal destructor
    // performs the existing final drain, which is allowed to complete pending
    // trace rows; the report itself never waits.
    session.reset();
    if (arguments.trace && recorder) {
        report_operation_trace(*recorder);
    }
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return execute(parse_arguments(argc, argv));
    } catch (const UsageError& error) {
        report("usage/input", error.what());
        return 2;
    } catch (const std::exception& error) {
        report_exception("execution", error);
        return 4;
    } catch (...) {
        report("execution", "unknown process failure");
        return 4;
    }
}
