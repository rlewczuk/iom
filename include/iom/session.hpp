#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "iom/chat_format.hpp"
#include "iom/device.hpp"
#include "iom/model.hpp"
#include "iom/token_selection.hpp"
#include "iom/tokenizer.hpp"

namespace iom {

namespace session_detail {
struct SessionAccess;
}

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
