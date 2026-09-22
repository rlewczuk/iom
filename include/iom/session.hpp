#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "iom/chat_format.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/model.hpp"
#include "iom/token_selection.hpp"
#include "iom/tokenizer.hpp"

namespace iom {

namespace session_detail {
struct SessionAccess;
}

/**
 * Why a low-level token generation request stopped.
 */
enum class GenerationStopReason { eos, max_new_tokens, context_capacity };

/**
 * Owned output of one low-level token generation request.
 */
struct TokenGenerationResult {
    std::vector<std::size_t> token_ids;
    GenerationStopReason stop_reason;
};

/**
 * Owning output of one high-level text or structured-chat generation request.
 */
struct GenerationResult {
    std::vector<std::size_t> token_ids;
    std::string text;
    GenerationStopReason stop_reason;
};

/**
 * Owning, reusable TinyLlama resource session.
 *
 * The session owns the realized model and every resource used by later
 * forward/generation stages.  The supplied Device is borrowed and must outlive
 * the session.  Resource setup is deliberately separate from model loading:
 * request preparation receives the exact workspace requirements computed by
 * the complete forward path and never estimates backend scratch.
 */
class TinyLlamaSession final {
public:
    ~TinyLlamaSession();

    TinyLlamaSession(const TinyLlamaSession&) = delete;
    TinyLlamaSession& operator=(const TinyLlamaSession&) = delete;
    TinyLlamaSession(TinyLlamaSession&&) = delete;
    TinyLlamaSession& operator=(TinyLlamaSession&&) = delete;

    [[nodiscard]] const TinyLlamaModel& model() const noexcept;
    [[nodiscard]] const TinyLlamaConfig& config() const noexcept;
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept;
    [[nodiscard]] const ChatFormatter& formatter() const noexcept;
    [[nodiscard]] TokenSelector& selector() noexcept;
    [[nodiscard]] const TokenSelector& selector() const noexcept;
    [[nodiscard]] Device& device() const noexcept;
    [[nodiscard]] DeviceOps& queue() noexcept;
    [[nodiscard]] const DeviceOps& queue() const noexcept;

    /**
     * Provision one exact logical request run.  `run_length` is the prompt or
     * prefill row count.  `operation` is the caller's checked maximum over all
     * actual operation and transfer requirements; it is not recomputed or
     * guessed here.  Selector host/device scratch is provisioned separately
     * from the selector's exact requirement.
     *
     * A second call first drains every accepted OID from the current request.
     * If any wait reports a retained failure, the old request remains owned and
     * the session stays poisoned; no replacement is published.
     */
    void prepare_request(
            std::size_t run_length, WorkspaceRequirements operation,
            TokenSelectorScratchRequirements selector_scratch);

    /**
     * Explicitly reserve this session's bounded operation trace before the
     * first request.  The call is valid only with an attached
     * `InferenceMetrics` recorder and only before the first request: an absent
     * recorder is rejected with `std::invalid_argument`, and a call after
     * request preparation is rejected with `std::logic_error`.  The
     * reservation reuses the request ledger's checked accepted-OID capacity
     * bound and may report `std::overflow_error` or `std::bad_alloc` before
     * any inference work exists; tracing is enabled only after the reservation
     * succeeds.  It performs no device work, adds no wait, and is not required
     * for scalar observation.
     */
    void prepare_operation_trace();
 
    /**
     * Run one synchronous greedy token-generation request from low-level token
     * IDs.  The returned token storage is owned by the result and remains
     * independent of later session reset or reuse.
     */
    [[nodiscard]] TokenGenerationResult generate_tokens(
            std::span<const std::size_t> token_ids,
            std::size_t max_new_tokens);

    /**
     * Generate from raw text using the session-owned tokenizer's automatic
     * special-token policy.
     */
    [[nodiscard]] GenerationResult generate_raw(
            std::string_view text, std::size_t max_new_tokens);

    /**
     * Render structured chat with a generation prompt, then tokenize it
     * without inserting automatic special tokens.
     */
    [[nodiscard]] GenerationResult generate_chat(
            std::span<const ChatMessageView> messages,
            std::size_t max_new_tokens);

    [[nodiscard]] std::size_t request_length() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;

private:
    friend struct session_detail::SessionAccess;
    struct Impl;

    explicit TinyLlamaSession(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
            const std::filesystem::path&, Device&);
    friend std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
            const std::filesystem::path&, Device&,
            std::unique_ptr<TokenSelector>, InferenceMetrics*);
};

[[nodiscard]] std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device);

/**
 * Load one session whose selector is supplied by the caller, optionally
 * attaching one borrowed, caller-owned observation recorder.  The recorder is
 * measured exactly once from this overload's entry through successful session
 * publication or the failed unwind; the caller-supplied selector is
 * constructed outside that interval, and outer device or allocator setup is
 * never part of it.  A null recorder leaves every observation hook disabled.
 * `load_tinyllama_session(path, device, nullptr)` is still the rejected
 * null-selector call and never means disabled instrumentation.
 */
[[nodiscard]] std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device,
        std::unique_ptr<TokenSelector> selector,
        InferenceMetrics* metrics = nullptr);

}  // namespace iom
