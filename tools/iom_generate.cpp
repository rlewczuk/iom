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

[[nodiscard]] int execute(const Arguments& arguments) {
    // Declaration order is deliberate: the CPU allocator outlives the device,
    // the device outlives the session, and the session outlives the result.
    ProcessAllocator cpu_allocator;
    std::unique_ptr<iom::Device> device;
    std::unique_ptr<iom::TinyLlamaSession> session;
    try {
        device = make_device(arguments, cpu_allocator);
        session = iom::load_tinyllama_session(
                std::filesystem::path(arguments.model_directory), *device);
    } catch (const std::exception& error) {
        report_exception("setup/load", error);
        return 3;
    } catch (...) {
        report("setup/load", "unknown setup failure");
        return 3;
    }

    std::vector<iom::ChatMessageView> message_views;
    std::optional<iom::GenerationResult> result;
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
        return 2;
    } catch (const std::exception& error) {
        report_exception("execution", error);
        return 4;
    } catch (...) {
        report("execution", "unknown generation failure");
        return 4;
    }

    try {
        std::cout.write(result->text.data(),
                        static_cast<std::streamsize>(result->text.size()));
        std::cout.flush();
        if (!std::cout) {
            throw std::runtime_error("decoded output write failed");
        }
    } catch (const std::exception& error) {
        report_exception("execution", error);
        return 4;
    } catch (...) {
        report("execution", "unknown output failure");
        return 4;
    }
    return 0;
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
