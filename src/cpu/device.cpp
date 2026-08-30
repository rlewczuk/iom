#include "iom/cpu/device.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iom/iom.hpp"

namespace iom {

    namespace {

        constexpr std::size_t kStorageAlignment = 32;

        // Section-3 host encoding: each logical element is one field of
        // exactly leaf_bits(type) bits, laid out least-significant bit
        // first at bit offset (element index) * bits. Multi-byte fields
        // are byte-aligned little-endian, which is the same bit stream.
        std::uint64_t read_bits(
                const unsigned char* base, std::size_t bit_offset,
                std::size_t nbits) {
            std::uint64_t value = 0;
            for (std::size_t i = 0; i < nbits; ++i) {
                const std::size_t bit = bit_offset + i;
                value |= static_cast<std::uint64_t>(
                                 (base[bit / 8] >> (bit % 8)) & 1)
                         << i;
            }
            return value;
        }

        void write_bits(
                unsigned char* base, std::size_t bit_offset,
                std::size_t nbits, std::uint64_t value) {
            for (std::size_t i = 0; i < nbits; ++i) {
                const std::size_t bit = bit_offset + i;
                unsigned char& byte = base[bit / 8];
                const unsigned char mask =
                        static_cast<unsigned char>(1u << (bit % 8));
                if ((value >> i) & 1) {
                    byte |= mask;
                } else {
                    byte &= static_cast<unsigned char>(~mask);
                }
            }
        }

        // Copies one encoded field between two bit-addressed byte buffers.
        // Every storage and host buffer carries the identical encoding, so
        // byte-aligned fields move as whole bytes and sub-byte fields move
        // bit by bit. No numeric conversion happens anywhere.
        void copy_value(
                unsigned char* destination, std::size_t destination_bit,
                const unsigned char* source, std::size_t source_bit,
                std::size_t nbits) {
            if (nbits % 8 == 0) {
                unsigned char* destination_bytes =
                        destination + destination_bit / 8;
                const unsigned char* source_bytes = source + source_bit / 8;
                if (destination_bytes != source_bytes) {
                    std::memcpy(destination_bytes, source_bytes, nbits / 8);
                }
                return;
            }
            write_bits(
                    destination, destination_bit, nbits,
                    read_bits(source, source_bit, nbits));
        }

        // Calls op(owner_plane, row, column, linear_index) for every
        // logical element of the view in row-major coordinate order. The
        // view's plane offset and strides carry the leading coordinates.
        template <typename Op>
        void for_each_coordinate(const TensorView& view, Op&& op) {
            const TensorSpec& spec = view.spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::size_t rows = dimensions[leading_rank];
            const std::size_t columns = dimensions[leading_rank + 1];
            const std::size_t count = spec.shape.element_count();
            const std::span<const std::size_t> strides =
                    view.plane_strides();

            for (std::size_t linear = 0; linear < count; ++linear) {
                std::size_t rest = linear;
                const std::size_t column = rest % columns;
                rest /= columns;
                const std::size_t row = rest % rows;
                rest /= rows;

                // View transforms keep every addressed owner plane in
                // bounds, so this accumulation cannot overflow.
                std::size_t plane = view.plane_offset();
                for (std::size_t k = leading_rank; k-- > 0;) {
                    plane += (rest % dimensions[k]) * strides[k];
                    rest /= dimensions[k];
                }
                op(plane, row, column, linear);
            }
        }

        // Owner plane of the view's linear_index-th logical element; the
        // row-major inverse of for_each_coordinate's plane accumulation.
        std::size_t plane_at(const TensorView& view, std::size_t linear) {
            const TensorSpec& spec = view.spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::size_t rows = dimensions[leading_rank];
            const std::size_t columns = dimensions[leading_rank + 1];
            const std::span<const std::size_t> strides =
                    view.plane_strides();

            std::size_t rest = linear / (rows * columns);
            std::size_t plane = view.plane_offset();
            for (std::size_t k = leading_rank; k-- > 0;) {
                plane += (rest % dimensions[k]) * strides[k];
                rest /= dimensions[k];
            }
            return plane;
        }

    }  // namespace

    class CpuDevice final : public Device {
    public:
        explicit CpuDevice(Allocator& allocator)
                : allocator_(allocator) {}

        [[nodiscard]] BackendKind backend_kind() const noexcept override {
            return BackendKind::CPU;
        }

        [[nodiscard]] std::uint32_t backend_device() const noexcept override {
            return 0;
        }

        [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                const TensorSpec& spec) override;
        [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

    private:
        Allocator& allocator_;
    };

    /**
     * Owner of one standard-layout allocation. The Tensor base validates
     * the specification before the body allocates; every construction
     * failure after allocation frees the storage before propagating.
     */
    class CpuTensor final : public Tensor {
    public:
        CpuTensor(const TensorSpec& spec, CpuDevice& device,
                  Allocator& allocator)
                : Tensor(spec, device),
                  allocator_(allocator) {
            void* address =
                    allocator_.alloc(view().spec().tiled_storage_nbytes());
            if (address == nullptr) {
                throw std::bad_alloc();
            }
            if (reinterpret_cast<std::uintptr_t>(address)
                            % kStorageAlignment
                    != 0) {
                allocator_.free(address);
                throw std::runtime_error(
                    "CPU tensor storage is not 32-byte aligned");
            }
            address_ = address;
        }

        ~CpuTensor() override {
            allocator_.free(address_);
        }

    private:
        [[nodiscard]] void* storage_handle() noexcept override {
            return address_;
        }

        void region_from_host(
                const TensorView& destination,
                std::span<const std::byte> source) override {
            unsigned char* storage =
                    static_cast<unsigned char*>(address_);
            const auto* host = reinterpret_cast<const unsigned char*>(
                    source.data());
            const std::size_t bits =
                    detail::leaf_bits(destination.spec().data_type);
            for_each_coordinate(
                    destination,
                    [&](std::size_t plane, std::size_t row,
                        std::size_t column, std::size_t linear) {
                        copy_value(
                                storage,
                                detail::standard_plane_slot(
                                        destination.spec(), plane, row, column)
                                        * bits,
                                host,
                                linear * bits,
                                bits);
                    });
        }

        void region_to_host(
                const TensorView& source,
                std::span<std::byte> destination) const override {
            // Zero first: padding never reaches the host buffer and unused
            // tail bits read as zero.
            std::fill(destination.begin(), destination.end(), std::byte{0});
            unsigned char* host =
                    reinterpret_cast<unsigned char*>(destination.data());
            const unsigned char* storage =
                    static_cast<const unsigned char*>(address_);
            const std::size_t bits =
                    detail::leaf_bits(source.spec().data_type);
            for_each_coordinate(
                    source,
                    [&](std::size_t plane, std::size_t row,
                        std::size_t column, std::size_t linear) {
                        copy_value(
                                host,
                                linear * bits,
                                storage,
                                detail::standard_plane_slot(
                                        source.spec(), plane, row, column)
                                        * bits,
                                bits);
                    });
        }

        Allocator& allocator_;
        void* address_ = nullptr;
    };

    /**
     * One in-order asynchronous copy queue with one worker thread. Tasks
     * are staged during submit() and handed to the worker only after
     * submit() returns, so a sequence is always committed in the common
     * base before the worker can report its completion.
     */
    class CpuQueue final : public DeviceOps {
    public:
        explicit CpuQueue(CpuDevice& device)
                : device_(&device),
                  worker_([this] { run(); }) {}

        ~CpuQueue() override {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
            }
            completion_.notify_all();
            worker_.join();
        }

        oid copy(const TensorView& source, TensorView& destination) override {
            validate_copy(source, destination);

            // Identical windows share owner storage, logical shape, plane
            // offset, and plane strides; equal native handles alone prove
            // nothing, so every component participates in the comparison.
            const bool identical_window =
                    source.native_handle() == destination.native_handle()
                    && source.plane_offset() == destination.plane_offset()
                    && std::equal(
                            source.plane_strides().begin(),
                            source.plane_strides().end(),
                            destination.plane_strides().begin(),
                            destination.plane_strides().end());

            const oid token = submit([&](std::uint64_t sequence) {
                std::lock_guard<std::mutex> lock(mutex_);
                staged_.push_back(
                        Task{sequence, &source, &destination,
                             identical_window});
            });
            publish_staged();
            return token;
        }

        oid add(const TensorView&, const TensorView&, TensorView&) override {
            throw unsupported("add");
        }

        oid mul(const TensorView&, const TensorView&, TensorView&) override {
            throw unsupported("mul");
        }

        oid silu(const TensorView&, TensorView&) override {
            throw unsupported("silu");
        }

        oid linear(const TensorView&, const TensorView&, TensorView&) override {
            throw unsupported("linear");
        }

        oid rmsnorm(const TensorView&, TensorView&, const TensorView&,
                    float, size_t) override {
            throw unsupported("rmsnorm");
        }

        oid sdpa(const TensorView&, const TensorView&, const TensorView&,
                 size_t, size_t, size_t, TensorView&) override {
            throw unsupported("sdpa");
        }

    private:
        struct Task {
            std::uint64_t sequence;
            const TensorView* source;
            TensorView* destination;
            bool no_op;
        };

        static std::runtime_error unsupported(const char* operation) {
            return std::runtime_error(
                    std::string("CPU backend does not implement ")
                    + operation);
        }

        void validate_copy(
                const TensorView& source,
                const TensorView& destination) const {
            if (&source.device() != device_
                    || &destination.device() != device_) {
                throw std::invalid_argument(
                    "copy views must belong to the queue's own device");
            }
            if (!(source.spec() == destination.spec())) {
                throw std::invalid_argument(
                    "copy views must have identical shape, leaf type, and "
                    "quantization");
            }
        }

        // Hands staged tasks to the worker. Called only after submit()
        // returned, which is what makes every staged sequence committed
        // before the worker can complete it.
        void publish_staged() {
            std::exception_ptr failure;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                while (!staged_.empty()) {
                    Task task = std::move(staged_.front());
                    staged_.pop_front();
                    try {
                        tasks_.push_back(std::move(task));
                    } catch (...) {
                        // The sequence is already consumed; report a failed
                        // completion so its wait cannot hang.
                        complete(task.sequence, std::current_exception());
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                }
                completion_.notify_one();
            }
            if (failure) {
                std::rethrow_exception(failure);
            }
        }

        void run() {
            std::unique_lock<std::mutex> lock(mutex_);
            for (;;) {
                completion_.wait(lock, [this] {
                    return shutdown_ || !tasks_.empty();
                });
                if (shutdown_) {
                    // Destruction neither waits for nor cancels submitted
                    // work: a contract-obeying caller has already waited
                    // for every task, so only unreachable leftovers could
                    // remain and their views may be gone.
                    return;
                }
                const Task task = std::move(tasks_.front());
                tasks_.pop_front();
                lock.unlock();

                std::exception_ptr failure;
                try {
                    if (!task.no_op) {
                        copy_elements(*task.source, *task.destination);
                    }
                } catch (...) {
                    failure = std::current_exception();
                }
                complete(task.sequence, failure);

                lock.lock();
            }
        }

        static void copy_elements(
                const TensorView& source, TensorView& destination) {
            const auto* source_base = static_cast<const unsigned char*>(
                    source.native_handle());
            auto* destination_base = static_cast<unsigned char*>(
                    destination.native_handle());
            const std::size_t bits =
                    detail::leaf_bits(source.spec().data_type);
            for_each_coordinate(
                    source,
                    [&](std::size_t source_plane, std::size_t row,
                        std::size_t column, std::size_t linear) {
                        const std::size_t destination_plane =
                                plane_at(destination, linear);
                        copy_value(
                                destination_base,
                                detail::standard_plane_slot(
                                        destination.spec(), destination_plane,
                                        row, column)
                                        * bits,
                                source_base,
                                detail::standard_plane_slot(
                                        source.spec(), source_plane, row,
                                        column)
                                        * bits,
                                bits);
                    });
        }

        CpuDevice* device_;
        std::mutex mutex_;
        std::condition_variable completion_;
        std::deque<Task> staged_;
        std::deque<Task> tasks_;
        bool shutdown_ = false;
        std::thread worker_;
    };

    std::unique_ptr<Tensor> CpuDevice::create_tensor(const TensorSpec& spec) {
        return std::make_unique<CpuTensor>(spec, *this, allocator_);
    }

    std::unique_ptr<DeviceOps> CpuDevice::create_ops() {
        return std::make_unique<CpuQueue>(*this);
    }

    std::unique_ptr<Device> make_cpu_device(Allocator& allocator) {
        return std::make_unique<CpuDevice>(allocator);
    }

}  // namespace iom
