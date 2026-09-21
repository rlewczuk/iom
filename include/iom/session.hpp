#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "iom/chat_format.hpp"
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
     * Run one synchronous greedy token-generation request from low-level token
     * IDs.  The returned token storage is owned by the result and remains
     * independent of later session reset or reuse.
     */
    [[nodiscard]] TokenGenerationResult generate_tokens(
            std::span<const std::size_t> token_ids,
            std::size_t max_new_tokens);

    /**
     * Generate from raw text using the session-owned tokenizer directly.
     */
    [[nodiscard]] GenerationResult generate_raw(
            std::string_view text, std::size_t max_new_tokens);

    /**
     * Render structured chat with a generation prompt, then tokenize it.
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
            std::unique_ptr<TokenSelector>);
};

[[nodiscard]] std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device);

[[nodiscard]] std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device,
        std::unique_ptr<TokenSelector> selector);

}  // namespace iom
