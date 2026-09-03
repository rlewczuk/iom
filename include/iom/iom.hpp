#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace iom {

    typedef uint64_t oid;

    /**
     * One in-order asynchronous operation queue over caller-created tensor
     * views. The common base leases one process-unique queue id in
     * [1, 255] at construction and releases it at destruction, allocates a
     * monotonic 56-bit submission sequence for every successfully queued
     * operation, validates wait tokens, and records completions so waits
     * are repeatable. The live-id pool is the only global queue state: it
     * never selects a backend, device, or runtime context. Calls on one
     * queue are serialized by the caller.
     */
    class DeviceOps {
    public:
        DeviceOps(const DeviceOps&) = delete;
        DeviceOps& operator=(const DeviceOps&) = delete;
        DeviceOps(DeviceOps&&) = delete;
        DeviceOps& operator=(DeviceOps&&) = delete;

        virtual ~DeviceOps();

        /**
         * Blocks until the token's operation completes, then keeps
         * succeeding on every later call; a retained asynchronous failure
         * is rethrown on every later call. Token zero, a zero sequence, a
         * live foreign queue id, and a sequence this queue never submitted
         * throw std::invalid_argument.
         */
        void wait(oid token);

        virtual oid copy(const TensorView& source, TensorView& destination) = 0;

        /** Addition: c = a + b **/
        virtual oid add(const TensorView& a, const TensorView& b, TensorView& c) = 0;

        /** Multiplication: c = a * b **/
        virtual oid mul(const TensorView& a, const TensorView& b, TensorView& c) = 0;

        /** SILU: y = silu(x) **/
        virtual oid silu(const TensorView& x, TensorView& y) = 0;

        /** Linear: y = x * w **/
        virtual oid linear(const TensorView& x, const TensorView& w, TensorView& y) = 0;

        /** RMSNorm: y = x * (1 / sqrt(mean(x^2, dim) + eps)) **/
        virtual oid rmsnorm(const TensorView& x, TensorView& y, const TensorView& w, float eps, size_t dim) = 0;

        /**
         * SDPA variant for GQA. We avoid repeating across k and v, kernel takes it into account automatically.
         * It always uses implicit casual mask.
         * Result is already rearranged so that it can be passed directly into linear projection o_proj.
         */
        virtual oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
                         size_t n_heads, size_t n_kv_heads, size_t head_dim, TensorView& attn_out) = 0;

    protected:
        DeviceOps();

        /**
         * Allocates this queue's next submission sequence, hands it to
         * queue_work, and returns the waitable token
         * (queue_id << 56) | sequence. The sequence is consumed only when
         * queue_work returns: a validation or synchronous pre-queue
         * failure propagates without burning a sequence number. Allocating
         * a sequence past 2^56 - 1 throws std::overflow_error before
         * queue_work runs.
         */
        template <typename QueueWork>
        oid submit(QueueWork queue_work) {
            std::uint64_t sequence = 0;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (next_sequence_ > kMaxSequence) {
                    throw std::overflow_error(
                        "DeviceOps 56-bit submission sequence is exhausted");
                }
                sequence = next_sequence_;
            }
            queue_work(sequence);
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                ++next_sequence_;
            }
            return encode_token(sequence);
        }

        /**
         * Records the in-order completion of sequence, and therefore of
         * every earlier submitted sequence. failure, when set, is retained
         * for exactly this sequence and rethrown by every later wait for
         * it. Wakes every waiter.
         */
        void complete(std::uint64_t sequence, std::exception_ptr failure = {});
        /**
         * Retains a failure for the sequence currently being submitted.
         * Unlike complete(), this does not mark the sequence complete: the
         * backend worker must still call complete() after its event fence
         * drains so callers cannot release operands early.
         */
        void commit_failure(
                std::uint64_t sequence, std::exception_ptr failure);

        /**
         * Test seam: moves the next allocated sequence forward without
         * submitting work, letting deterministic fakes begin near the
         * 56-bit limit. Never moves backward and never allocates.
         */
        void seek_next_sequence(std::uint64_t next_sequence);

    private:
        [[nodiscard]] static std::uint8_t lease_queue_id();
        static void release_queue_id(std::uint8_t queue_id) noexcept;
        [[nodiscard]] oid encode_token(std::uint64_t sequence) const noexcept;

        static constexpr std::uint64_t kSequenceBits = 56;
        static constexpr std::uint64_t kSequenceMask = (std::uint64_t{1} << kSequenceBits) - 1;
        static constexpr std::uint64_t kMaxSequence = kSequenceMask;

        std::uint8_t queue_id_;
        std::uint64_t next_sequence_ = 1;
        std::uint64_t completed_ = 0;
        std::mutex completion_mutex_;
        std::condition_variable completion_cv_;
        std::map<std::uint64_t, std::exception_ptr> failures_;
        std::map<std::uint64_t, std::exception_ptr> pending_failures_;
    };

    class Block {
    public:
        virtual ~Block() = default;
        virtual void forward(const Tensor& x, Tensor& y) = 0;
    };
}
