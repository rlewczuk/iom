#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <initializer_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <span>

#include "detail/outstanding_work_registry.hpp"
#include "detail/staged_worker.hpp"
#include "oid.hpp"
#include "tensor.hpp"

namespace iom {

    namespace detail {
        struct CheckedViewFacts;
        // A backend dispatch may defer an accepted FIFO head when a fixed
        // completion or metadata resource is quarantined. This is not a
        // terminal operation failure and must leave the node parked.
        class AdmissionResourceUnavailable final : public std::exception {
        public:
            [[nodiscard]] const char* what() const noexcept override {
                return "fixed admission resource is unavailable";
            }
        };
    }  // namespace detail

    namespace detail {
        /**
         * Shared reusable raw-workspace validation (leaf 05). Centralizes
         * the required-capacity, base/range alignment, checked-offset,
         * exact-Device-identity, exact-live-owner-identity, and
         * workspace-versus-operand/output overlap rules that tasks 07 and
         * 11 enforce on borrowed workspace arguments. Every rejection is
         * invalid input: `std::invalid_argument`; range-end overflow
         * reports `std::overflow_error`. Pure: no allocation, registry,
         * lease, token/queue resource, or backend effect.
         */
        class WorkspaceValidation {
        public:
            /**
             * Validates `workspace` as the scratch range for one operation
             * on `device` needing `required_capacity` bytes at
             * `required_alignment`. A zero capacity accepts any view (the
             * facades default to the empty view exactly for this case); a
             * positive capacity additionally requires a non-empty view of
             * a live owner created by this exact device, sufficient
             * capacity, and the required alignment of its real address or
             * opaque owner-relative logical offset. Its backing must be
             * disjoint from every operand/output. Returns the unchanged view.
             */
            [[nodiscard]] static RawWorkspaceView validated(
                    const Device& device,
                    const RawWorkspaceView& workspace,
                    std::size_t required_capacity,
                    std::size_t required_alignment,
                    std::span<const TensorView> operands);
            [[nodiscard]] static void* address(
                    const RawWorkspaceView& workspace) noexcept;
            // Checked backing and logical extent of an admitted live view.
            [[nodiscard]] static StorageRange range(
                    const RawWorkspaceView& workspace);
        };

    }  // namespace detail
    /**
     * Explicit output mode of the linear projection. The mode is an
     * argument and is never inferred from the output rank: `ordinary`
     * writes `out[...,R,O]` and requires `H=1,D=O`, while `head_planar`
     * writes `out[...,H,R,D]` and requires the checked equality `O=H*D`.
     * Any other value is invalid input.
     */
    enum class LinearOutputLayout { ordinary, head_planar };
    /**
     * One in-order asynchronous operation queue over caller-created tensor
     * views. Every OID-returning operation is a common `noexcept` facade:
     * it validates, maps failures, encodes tokens, and registers lifetimes
     * before backend effects or token acceptance. Backend hooks cannot bypass
     * this protocol.
     *
     * `oid` is signed `int64_t`: -1 InvalidArgument, -2 Unsupported, -3
     * Overflow, -4 ResourceExhausted, -5 DeviceError, and -6 InternalError.
     * Negative results are synchronous errors, positive values are accepted
     * tokens, and zero is invalid. `add`, `mul`, `sub`, and `div` each have
     * exactly three views and are the binary operation support signals.
     * Every full tensor, view, and binary result shape must have rank two
     * through eight; a rank outside that interval is an invalid argument
     * and maps through OidError::InvalidArgument (OID -1) before any
     * sequence, token, registry entry, metadata upload, or backend effect.
     * Synchronous failures never cross the facade; `wait` throws for invalid
     * tokens and retained post-acceptance failures.
     */
    class DeviceOps {
    public:
        DeviceOps(const DeviceOps&) = delete;
        DeviceOps& operator=(const DeviceOps&) = delete;
        DeviceOps(DeviceOps&&) = delete;
        DeviceOps& operator=(DeviceOps&&) = delete;
        virtual ~DeviceOps();
        [[nodiscard]] const Device& device() const;
        /**
         * Observe accepted work in queue order. A negative, zero, foreign,
         * future, skipped/reserved-but-never-submitted, or otherwise
         * unsubmitted value throws std::invalid_argument immediately; a
         * skipped value remains invalid after later completion. Accepted
         * tokens are repeat-waitable, and retained asynchronous failures are
         * rethrown by every wait.
         */
        void wait(oid token);
        /**
         * Common `noexcept` OID facades for three-view binary operations.
         * Validation and lifetime ownership are shared by all operations.
         * All three operand/output full specs and the computed broadcast
         * result must have rank two through eight; invalid rank returns
         * OidError::InvalidArgument with no sequence, token, registry,
         * metadata, or backend effect.
         */
        oid copy(const TensorView& source, TensorView& destination) noexcept;
        oid add(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid mul(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid sub(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid div(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        /**
         * Pure deterministic raw-workspace requirement queries for the
         * three-view binary operations (leaf 05). Each runs the exact
         * validation of its binary operation — operation, all three
         * views, exact device identity, specifications, rank and view
         * bounds, broadcasting, alias rules, and backend capability —
         * and then reports, with no allocation, registration, lease,
         * token/queue resource, metadata upload, submission, and no
         * dependence on free data-arena capacity, fragmentation, queue
         * occupancy, or completion state:
         * CPU, CUDA, and ROCm report `{0, 1}`; SYCL reports its
         * checked whole-plane staging sum at alignment 32. Validation
         * failures surface as the corresponding exception
         * (`std::invalid_argument`, `std::overflow_error`, unsupported
         * operation), never through the OID error mapping.
         */
        [[nodiscard]] WorkspaceRequirements add_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements mul_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements sub_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements div_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        /**
         * Backend-neutral cache-row append:
         * `destination[..., h, a + r, d] = source[..., h, r, d]`.
         * The source and destination ranks are three through eight, the
         * leading tuple, head count, feature width, encodings, and exact
         * queue device must match, and the explicit row offset is never
         * inferred from session state.
         */
        oid cache_append(
                const TensorView& source, TensorView& destination,
                std::size_t a, RawWorkspaceView workspace = {}) noexcept;
        [[nodiscard]] WorkspaceRequirements
                cache_append_workspace_requirements(
                        const TensorView& source,
                        const TensorView& destination, std::size_t a);
        /**
         * Embedding row lookup `out[b, r, f] = table[indices[b, 0, r], f]`
         * over a rank-two `E[V, F]` table shared unchanged by every
         * independent leading plane, `indices[..., 1, R]`, and
         * `out[..., R, F]`. Index and output views carry the same leading
         * tuple; all three views keep rank two through eight, nonzero
         * runtime extents, and their own selected plane offset and leading
         * strides. There is no leading broadcast, singleton output-rank
         * inflation, final-axis transform, or padding-dependent behavior.
         * Table and output share one leaf encoding and
         * `QuantizationFormat::NONE`; the index leaf is one of the twelve
         * integral leaves, because `BOOL` and every floating leaf are not
         * ID semantics. Selected payload bits are copied unchanged, with no
         * arithmetic or re-encoding.
         *
         * Admission precedes any backend effect: well-formed views, rank
         * and shape structure, exact queue device identity, live owner and
         * stable native handle, leading bounds, checked element, byte,
         * tile, plane, and stride arithmetic, table/output encoding
         * agreement, and conservative output/input overlap. Input/input
         * read aliases are valid. A supplied workspace is validated only
         * for a positive backend requirement, after the capability query.
         * Negative results are synchronous errors and positive values are
         * accepted tokens whose retained failures rethrow on every wait.
         *
         * `embedding_workspace_requirements` is pure and deterministic: it
         * runs the same validation and consults the backend capability
         * without allocating, registering, leasing, consuming a sequence,
         * uploading metadata, reading an index, or submitting. Both backend
         * hooks default to `Unsupported`, so a backend advertises this
         * operation only by overriding them; the operation-owned Embedding
         * lookup section of `docs/BACKEND_CONTRACT.md` stays the normative
         * source for per-backend payload, index, workspace, status, and
         * failure policy.
         */
        oid embedding(const TensorView& table, const TensorView& indices,
                      TensorView& out,
                      RawWorkspaceView workspace = {}) noexcept;
        [[nodiscard]] WorkspaceRequirements embedding_workspace_requirements(
                const TensorView& table, const TensorView& indices,
                const TensorView& out);
        oid silu(const TensorView& x, TensorView& y,
                 RawWorkspaceView workspace = {}) noexcept;
        /**
         * Pure zero-workspace requirement query for SiLU. It validates both
         * views and invokes the backend capability hook with fixed-size
         * value snapshots, without allocating, registering owners, leasing
         * workspace, consuming a queue sequence, or submitting work.
         */
        [[nodiscard]] WorkspaceRequirements silu_workspace_requirements(
                const TensorView& x, const TensorView& y);
        /**
         * Bias-free blocked linear projection of independently selected
         * input rows. `x` is `[...,T,I]` with rank two through eight and
         * `w` is the rank-two Hugging Face `[O,I]` weight shared unchanged
         * by every independent leading plane: output coordinate `o`
         * selects weight row `w[o,*]`, so no checkpoint weight is ever
         * transposed. `LinearOutputLayout` selects the output layout
         * explicitly and is never inferred from the output rank.
         * `ordinary` writes `[...,R,O]` and requires `H=1,D=O`;
         * `head_planar` writes `[...,H,R,D]`, requires the checked equality
         * `O=H*D`, and inserts one head axis without exceeding rank eight.
         * The input and output leading tuples must match exactly after
         * excluding that inserted head axis; the rank-two weight has no
         * leading tuple and implies no leading-state broadcast.
         *
         * `s` selects the first source row and `R` the projected row count:
         * `R > 0`, `s <= T`, and `R <= T-s`, so `R` is never confused with,
         * inferred from, or wrapped around `T`, `s`, or `O`. `H` and `D`
         * are the head count and head width and are both nonzero. The two
         * equations are `out[b,r,o] = sum_i x[b,s+r,i] * w[o,i]` and
         * `out[b,h,r,d] = sum_i x[b,s+r,i] * w[h*D+d,i]`; only the newly
         * computed selected rows are written, and the final untied LM head
         * is ordinary mode with `s=T-1`, `R=1`, `H=1`, and `D=O`.
         *
         * Admission precedes any backend effect: nonzero extents and rank,
         * the exact output shape and layout relation, the leading tuple,
         * the selected row window, rank growth, exact queue device
         * identity, live owner and stable native handle, selected-plane
         * bounds and transformed strides, checked element, byte, address,
         * plane, tile, stride, and `H*D` arithmetic, then conservative
         * output/input overlap, one applicable leaf and
         * `QuantizationFormat::NONE` for all three views, and the immutable
         * backend capability. Read/read aliasing between `x` and `w` is
         * valid, and exact aliases deduplicate their owner registration;
         * output storage stays disjoint from both inputs and from the
         * workspace. A supplied workspace is validated only against the
         * requirement reported by `linear_workspace_requirements`, after
         * that capability query: a positive requirement needs a live,
         * sufficient, required-alignment range disjoint from every operand
         * and the output, while a zero requirement neither validates nor
         * leases an unused range. Negative results are synchronous errors
         * that consume no token, and positive values are accepted tokens
         * whose retained failures rethrow on every wait.
         *
         * `linear_workspace_requirements` is pure and deterministic: it
         * runs exactly the same validation and consults the same backend
         * capability without allocating, constructing a view snapshot or
         * request, registering an owner, acquiring a lease, consuming a
         * sequence, reading data, mutating queue state, or submitting, and
         * without depending on queue occupancy or completion state. Both
         * backend hooks default to `Unsupported`, so a backend advertises
         * this operation only by overriding them, and the operation-owned
         * Linear projections section of `docs/BACKEND_CONTRACT.md` remains
         * the normative source for per-backend payload, workspace, status,
         * and failure policy.
         */
        oid linear(const TensorView& x, const TensorView& w, TensorView& out,
                   std::size_t s, std::size_t R, LinearOutputLayout layout,
                   std::size_t H, std::size_t D,
                   RawWorkspaceView workspace = {}) noexcept;
        [[nodiscard]] WorkspaceRequirements linear_workspace_requirements(
                const TensorView& x, const TensorView& w,
                const TensorView& out, std::size_t s, std::size_t R,
                LinearOutputLayout layout, std::size_t H, std::size_t D);
        /**
         * Common `noexcept` facade for RMS normalization. `x` and `out` have
         * identical logical `[...,R,F]` shape with rank two through eight and
         * nonzero extents; rank-two `scale` is exactly `[1,F]` and is shared
         * across every leading plane and row. Each row reduces over its own
         * `F` logical features only, and every plane, row, padding element,
         * and other request stays independent. The redundant `dim` argument
         * is gone: the feature extent is the operation's `F`.
         *
         * Admission validates shape, device, owner, handle, stride and
         * checked range arithmetic, dtype and `QuantizationFormat::NONE`,
         * disjoint output storage, and finite nonnegative `eps` before the
         * immutable backend capability. A nonempty workspace is invalid,
         * because the queried requirement is exactly `{0, 1}`. Failures are
         * mapped to the established negative OIDs without consuming a
         * sequence, registering an owner, submitting work, or mutating
         * output; accepted work returns a positive token whose owners stay
         * registered through proven completion. The default backend hook is
         * `Unsupported` until an operation port replaces it.
         */
        oid rmsnorm(const TensorView& x, const TensorView& scale,
                    TensorView& out, float eps,
                    RawWorkspaceView workspace = {}) noexcept;
        /**
         * Pure deterministic raw-workspace requirement query for `rmsnorm`.
         * It runs exactly the admission validation of the operation — shape,
         * exact device identity, live owners and handles, leading bounds and
         * strides, checked arithmetic, dtype/quantization, disjoint output,
         * epsilon, and immutable backend capability — and then reports the
         * requirement with no allocation, request or vector snapshot
         * construction, owner or lease registration, sequence/token
         * consumption, data read, state mutation, or submission, and no
         * dependence on queue occupancy or completion state. Every supported
         * implementation returns exactly `{0, 1}` and consumes no
         * `RawWorkspace`, so the only admissible supplied workspace is the
         * empty default. Validation failures and an unsupported backend
         * capability surface as the corresponding exception
         * (`std::invalid_argument`, `std::overflow_error`,
         * `UnsupportedOperation`), never through OID error mapping.
         */
        [[nodiscard]] WorkspaceRequirements rmsnorm_workspace_requirements(
                const TensorView& x, const TensorView& scale,
                const TensorView& out, float eps);
        /**
         * Common `noexcept` facade for split-half rotary position encoding.
         * `x` and `out` have identical logical `[...,H,R,D]` shapes with
         * rank three through eight, independent leading-plane mappings,
         * nonzero `H` and `R`, and positive even `D`. `a` is the explicit
         * absolute position of the first row and `theta` is the explicit
         * runtime base. The common layer owns admission, immutable metadata
         * snapshots, OID error mapping, and completion lifetime; an
         * unported backend remains `Unsupported`.
         */
        oid rope(const TensorView& x, TensorView& out, std::size_t a,
                 double theta,
                 RawWorkspaceView workspace = {}) noexcept;
        /**
         * Pure deterministic workspace query for `rope`. It accepts no
         * workspace and performs the same allocation-free admission and
         * capability validation as submission. A supported implementation
         * reports exactly `{0, 1}`; an unported backend throws
         * `detail::UnsupportedOperation`.
         */
        [[nodiscard]] WorkspaceRequirements rope_workspace_requirements(
                const TensorView& x, const TensorView& out, std::size_t a,
                double theta);

        oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
                 TensorView& out, std::size_t a, std::size_t L,
                 RawWorkspaceView workspace = {}) noexcept;
        [[nodiscard]] WorkspaceRequirements sdpa_workspace_requirements(
                const TensorView& q, const TensorView& k, const TensorView& v,
                const TensorView& out, std::size_t a, std::size_t L);

    protected:
        /*
         * Backend hooks are implementation extension points only. Public
         * facades retain common validation, error mapping, token encoding,
         * queue ordering, and lifetime registration.
         */
        enum class BinaryOperation { Add, Mul, Sub, Div };
        struct BinaryViewSnapshot {
            TensorSpec spec;
            const Device* device_identity;
            const Tensor* owner_identity;
            void* native_handle;
            std::size_t plane_offset;
            std::vector<std::size_t> plane_strides;
            std::vector<std::size_t> logical_plane_strides;
            bool broadcast_rows;
            bool broadcast_columns;
            bool broadcasts;
        };
        struct CopyViewSnapshot {
            TensorSpec spec;
            const Device* device_identity;
            const Tensor* owner_identity;
            void* native_handle;
            std::size_t plane_offset;
            std::vector<std::size_t> plane_strides;
        };
        /**
         * Fixed-size, immutable metadata captured for one cache append
         * operand. The arrays deliberately avoid TensorSpec/TensorShape
         * copies so a successful workspace query does not allocate.
         */
        struct CacheAppendViewSnapshot {
            std::size_t rank = 0;
            std::array<std::size_t, 8> dimensions{};
            std::array<std::size_t, 6> plane_strides{};
            std::size_t plane_offset = 0;
            DataType data_type = DataType::BOOL;
            QuantizationFormat quantization = QuantizationFormat::NONE;
            const Device* device_identity = nullptr;
            const Tensor* owner_identity = nullptr;
            void* native_handle = nullptr;

            [[nodiscard]] std::span<const std::size_t>
                    shape_dimensions() const noexcept {
                return {dimensions.data(), rank};
            }
            [[nodiscard]] std::span<const std::size_t>
                    leading_plane_strides() const noexcept {
                return {plane_strides.data(), rank >= 2 ? rank - 2 : 0};
            }
        };
        /**
         * Immutable cache append admission request. Source/destination
         * metadata and `a` are value-copied before backend admission; only
         * the validated workspace and retained lease are populated while
         * the common submission helper prepares the accepted work.
         */
        struct CacheAppendRequest {
            CacheAppendViewSnapshot source;
            CacheAppendViewSnapshot destination;
            std::size_t a = 0;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;

            CacheAppendRequest(
                    CacheAppendViewSnapshot source_,
                    CacheAppendViewSnapshot destination_,
                    std::size_t a_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : source(std::move(source_)),
                  destination(std::move(destination_)), a(a_),
                  workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            CacheAppendRequest(const CacheAppendRequest&) = default;
            CacheAppendRequest& operator=(const CacheAppendRequest&) = delete;
            CacheAppendRequest(CacheAppendRequest&& other) noexcept
                : source(std::move(other.source)),
                  destination(std::move(other.destination)), a(other.a),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };
        /**
         * Immutable admission snapshot of one linear projection operand:
         * value-copied view metadata and the stable device, owner, and
         * native-handle identities of that view. No callback may retain the
         * caller's borrowed `TensorView` object, and snapshot values do not
         * change when caller views or their backing metadata are mutated or
         * destroyed.
         */
        struct LinearViewSnapshot {
            TensorSpec spec;
            const Device* device_identity;
            const Tensor* owner_identity;
            void* native_handle;
            std::size_t plane_offset;
            std::vector<std::size_t> plane_strides;
        };
        [[nodiscard]] static CopyViewSnapshot snapshot_copy_view(
                const TensorView& view);
        struct BinaryRequest {
            BinaryOperation operation;
            BinaryViewSnapshot lhs;
            BinaryViewSnapshot rhs;
            BinaryViewSnapshot out;
            TensorShape result_shape;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;
            BinaryRequest(
                    BinaryOperation operation_,
                    BinaryViewSnapshot lhs_, BinaryViewSnapshot rhs_,
                    BinaryViewSnapshot out_, TensorShape result_shape_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : operation(operation_), lhs(std::move(lhs_)),
                  rhs(std::move(rhs_)), out(std::move(out_)),
                  result_shape(std::move(result_shape_)),
                  workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            BinaryRequest(const BinaryRequest&) = default;
            BinaryRequest& operator=(const BinaryRequest&) = delete;
            BinaryRequest(BinaryRequest&& other) noexcept
                : operation(other.operation), lhs(std::move(other.lhs)),
                  rhs(std::move(other.rhs)), out(std::move(other.out)),
                  result_shape(std::move(other.result_shape)),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };

        /**
         * Immutable admission snapshot of one embedding submission:
         * value-copied view metadata and stable owner identities for the
         * table, index, and output operands, the validated caller workspace,
         * the backend-reported requirement, and the workspace lease retained
         * through proven completion. No callback may retain the caller's
         * borrowed `TensorView` object, and no field changes after
         * admission.
         */
        struct EmbeddingRequest {
            CopyViewSnapshot table;
            CopyViewSnapshot indices;
            CopyViewSnapshot out;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;
            EmbeddingRequest(
                    CopyViewSnapshot table_, CopyViewSnapshot indices_,
                    CopyViewSnapshot out_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : table(std::move(table_)), indices(std::move(indices_)),
                  out(std::move(out_)), workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            EmbeddingRequest(const EmbeddingRequest&) = default;
            EmbeddingRequest& operator=(const EmbeddingRequest&) = delete;
            EmbeddingRequest(EmbeddingRequest&& other) noexcept
                : table(std::move(other.table)),
                  indices(std::move(other.indices)), out(std::move(other.out)),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };

        /**
         * Immutable admission snapshot of one linear projection
         * submission: value-copied view metadata and stable owner
         * identities for the input, weight, and output operands, the
         * validated row window `start_row`/`rows`, output mode `layout`,
         * head scalars `heads`/`head_dim`, the validated caller workspace,
         * the backend-reported requirement, and the workspace lease
         * retained through proven completion. No callback may retain the
         * caller's borrowed `TensorView` object, and no field changes after
         * admission.
         */
        struct LinearRequest {
            LinearViewSnapshot x;
            LinearViewSnapshot w;
            LinearViewSnapshot out;
            std::size_t start_row;
            std::size_t rows;
            LinearOutputLayout layout;
            std::size_t heads;
            std::size_t head_dim;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;
            LinearRequest(
                    LinearViewSnapshot x_, LinearViewSnapshot w_,
                    LinearViewSnapshot out_, std::size_t start_row_,
                    std::size_t rows_, LinearOutputLayout layout_,
                    std::size_t heads_, std::size_t head_dim_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : x(std::move(x_)), w(std::move(w_)), out(std::move(out_)),
                  start_row(start_row_), rows(rows_), layout(layout_),
                  heads(heads_), head_dim(head_dim_), workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            LinearRequest(const LinearRequest&) = default;
            LinearRequest& operator=(const LinearRequest&) = delete;
            LinearRequest(LinearRequest&& other) noexcept
                : x(std::move(other.x)), w(std::move(other.w)),
                  out(std::move(other.out)), start_row(other.start_row),
                  rows(other.rows), layout(other.layout),
                  heads(other.heads), head_dim(other.head_dim),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };

        // Immutable host-side copy descriptor retained by admission. It
        // contains value-copied view metadata and stable owner identities; no
        // callback may retain the caller's borrowed view object.
        struct CopyRequest {
            CopyViewSnapshot source;
            CopyViewSnapshot destination;
            bool no_op = false;
            CopyRequest(
                    const TensorView& source_, const TensorView& destination_,
                    bool no_op_)
                : source(snapshot_copy_view(source_)),
                  destination(snapshot_copy_view(destination_)),
                  no_op(no_op_) {}
        };

        // Immutable host-side RMS normalization request retained by
        // admission. It owns value-copied view metadata — shape, leaf type,
        // quantization, plane offset, and plane strides — and the stable
        // device, owner, and native-handle identities of `x`, `scale`, and
        // `out`, plus the validated epsilon and the admitted workspace
        // requirement. No callback may retain the caller's borrowed view
        // object, and snapshot values do not change when caller views or
        // their backing metadata are mutated or destroyed.
        struct RmsnormRequest {
            CopyViewSnapshot x;
            CopyViewSnapshot scale;
            CopyViewSnapshot out;
            float epsilon;
            WorkspaceRequirements workspace_requirements;
            RmsnormRequest(
                    CopyViewSnapshot x_, CopyViewSnapshot scale_,
                    CopyViewSnapshot out_, float epsilon_,
                    WorkspaceRequirements workspace_requirements_ = {})
                : x(std::move(x_)), scale(std::move(scale_)),
                  out(std::move(out_)), epsilon(epsilon_),
                  workspace_requirements(workspace_requirements_) {}
            RmsnormRequest(const RmsnormRequest&) = default;
            RmsnormRequest& operator=(const RmsnormRequest&) = delete;
            RmsnormRequest(RmsnormRequest&& other) noexcept
                : x(std::move(other.x)), scale(std::move(other.scale)),
                  out(std::move(other.out)), epsilon(other.epsilon),
                  workspace_requirements(other.workspace_requirements) {}
        };
        /**
         * Fixed-capacity immutable admission snapshot for one RoPE view.
         * Every scalar and array is copied from the caller's view; no
         * `TensorView`, dynamic shape, or borrowed metadata survives the
         * facade call. The checked facts are retained so backend adapters can
         * address the admitted view without repeating unchecked arithmetic.
         */
        struct RopeViewSnapshot {
            std::size_t rank = 0;
            std::array<std::size_t, 8> dimensions{};
            std::array<std::size_t, 6> plane_strides{};
            std::size_t plane_offset = 0;
            DataType data_type = DataType::BOOL;
            QuantizationFormat quantization = QuantizationFormat::NONE;
            const Device* device_identity = nullptr;
            const Tensor* owner_identity = nullptr;
            void* native_handle = nullptr;
            std::size_t max_plane = 0;
            std::size_t addressed_bytes = 0;
            std::size_t storage_bytes = 0;
            std::size_t logical_bytes = 0;

            [[nodiscard]] std::span<const std::size_t>
                    shape_dimensions() const noexcept {
                return {dimensions.data(), rank};
            }
            [[nodiscard]] std::span<const std::size_t>
                    leading_plane_strides() const noexcept {
                return {plane_strides.data(), rank >= 2 ? rank - 2 : 0};
            }
        };

        /**
         * Immutable host-side RoPE request retained by accepted work. It
         * carries fixed-capacity value snapshots, explicit absolute position
         * and theta, the validated zero-workspace view and exact requirement,
         * plus the lease populated by the prepared submission helper.
         */
        struct RopeRequest {
            RopeViewSnapshot x;
            RopeViewSnapshot out;
            std::size_t a;
            double theta;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;
            RopeRequest(
                    RopeViewSnapshot x_, RopeViewSnapshot out_,
                    std::size_t a_, double theta_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : x(std::move(x_)), out(std::move(out_)), a(a_),
                  theta(theta_), workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            RopeRequest(const RopeRequest&) = default;
            RopeRequest& operator=(const RopeRequest&) = delete;
            RopeRequest(RopeRequest&& other) noexcept
                : x(std::move(other.x)), out(std::move(other.out)),
                  a(other.a), theta(other.theta),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };

        /**
         * Fixed-capacity immutable metadata for one SDPA operand. The
         * dimensions, plane map, selected offset, checked bounds, and stable
         * owner identities are copied before a request reaches a backend
         * hook. Keeping the shape in bounded arrays makes the requirement
         * query allocation-free while retaining the same facts for deferred
         * submission.
         */
        struct SdpaViewSnapshot {
            std::size_t rank = 0;
            std::array<std::size_t, 8> dimensions{};
            std::array<std::size_t, 6> plane_strides{};
            std::size_t plane_offset = 0;
            DataType data_type = DataType::BOOL;
            QuantizationFormat quantization = QuantizationFormat::NONE;
            const Device* device_identity = nullptr;
            const Tensor* owner_identity = nullptr;
            void* native_handle = nullptr;
            std::size_t max_plane = 0;
            std::size_t addressed_bytes = 0;
            std::size_t storage_bytes = 0;
            std::size_t logical_bytes = 0;

            [[nodiscard]] std::span<const std::size_t>
                    shape_dimensions() const noexcept {
                return {dimensions.data(), rank};
            }
            [[nodiscard]] std::span<const std::size_t>
                    leading_plane_strides() const noexcept {
                return {plane_strides.data(), rank >= 2 ? rank - 2 : 0};
            }
        };

        /**
         * Immutable, value-owned SDPA admission request. All operation
         * scalars and four fixed-capacity view snapshots survive caller view
         * mutation/destruction; workspace and its lease are populated only
         * after the backend requirement hook succeeds.
         */
        struct SdpaRequest {
            SdpaViewSnapshot q;
            SdpaViewSnapshot k;
            SdpaViewSnapshot v;
            SdpaViewSnapshot out;
            std::size_t a = 0;
            std::size_t L = 0;
            std::size_t Hq = 0;
            std::size_t Hkv = 0;
            std::size_t R = 0;
            std::size_t C = 0;
            std::size_t D = 0;
            std::size_t grouping = 0;
            std::size_t output_width = 0;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;

            SdpaRequest(
                    SdpaViewSnapshot q_, SdpaViewSnapshot k_,
                    SdpaViewSnapshot v_, SdpaViewSnapshot out_,
                    std::size_t a_, std::size_t L_, std::size_t Hq_,
                    std::size_t Hkv_, std::size_t R_, std::size_t C_,
                    std::size_t D_, std::size_t grouping_,
                    std::size_t output_width_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : q(std::move(q_)), k(std::move(k_)), v(std::move(v_)),
                  out(std::move(out_)), a(a_), L(L_), Hq(Hq_), Hkv(Hkv_),
                  R(R_), C(C_), D(D_), grouping(grouping_),
                  output_width(output_width_), workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            SdpaRequest(const SdpaRequest&) = default;
            SdpaRequest& operator=(const SdpaRequest&) = delete;
            SdpaRequest(SdpaRequest&& other) noexcept
                : q(std::move(other.q)), k(std::move(other.k)),
                  v(std::move(other.v)), out(std::move(other.out)),
                  a(other.a), L(other.L), Hq(other.Hq), Hkv(other.Hkv),
                  R(other.R), C(other.C), D(other.D),
                  grouping(other.grouping), output_width(other.output_width),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };


        /**
         * Fixed-size immutable SiLU view metadata captured before admission.
         * The bounded arrays avoid heap allocation in the pure requirement
         * query while preserving every rank, extent, leading plane stride,
         * selected plane, encoding, and stable owner identity needed by a
         * deferred backend callback.
         */
        struct SiLUViewSnapshot {
            std::size_t rank = 0;
            std::array<std::size_t, 8> dimensions{};
            std::array<std::size_t, 8> plane_strides{};
            std::size_t plane_offset = 0;
            DataType data_type = DataType::BOOL;
            QuantizationFormat quantization = QuantizationFormat::NONE;
            const Device* device_identity = nullptr;
            const Tensor* owner_identity = nullptr;
            void* native_handle = nullptr;

            [[nodiscard]] std::span<const std::size_t>
                    shape_dimensions() const noexcept {
                return {dimensions.data(), rank};
            }
            [[nodiscard]] std::span<const std::size_t>
                    leading_plane_strides() const noexcept {
                const std::size_t leading =
                        rank >= 2 ? rank - 2 : 0;
                return {plane_strides.data(), leading};
            }
        };

        /**
         * Immutable, value-copied SiLU admission request. The caller's
         * borrowed TensorView and RawWorkspaceView never reach a callback;
         * zero-capacity workspace is represented solely by the copied
         * requirement.
         */
        struct SiLURequest {
            SiLUViewSnapshot x;
            SiLUViewSnapshot y;
            WorkspaceRequirements workspace_requirements{0, 1};

            SiLURequest(
                    SiLUViewSnapshot x_, SiLUViewSnapshot y_,
                    WorkspaceRequirements workspace_requirements_ = {0, 1})
                : x(std::move(x_)), y(std::move(y_)),
                  workspace_requirements(workspace_requirements_) {}
            SiLURequest(const SiLURequest&) = default;
            SiLURequest& operator=(const SiLURequest&) = delete;
            SiLURequest(SiLURequest&&) noexcept = default;
        };

        DeviceOps();
        explicit DeviceOps(const Device& device);
        virtual oid copy_impl(
                const TensorView& source, TensorView& destination);
        virtual oid binary_impl(const BinaryRequest& request);
        virtual oid cache_append_impl(const CacheAppendRequest& request);
        virtual oid embedding_impl(const EmbeddingRequest& request);
        virtual oid silu_impl(const SiLURequest& request);
        [[nodiscard]] virtual WorkspaceRequirements
                silu_workspace_requirements_impl(
                        const SiLURequest& request);
        /**
         * Immutable linear projection execution hook. It receives the
         * already validated, admission-snapshotted request and queues the
         * backend's own kernel for it; the default common implementation
         * reports every request `Unsupported`, so a valid-shape linear
         * request on an unported backend stays `Unsupported` before owner
         * registration, sequence consumption, dispatch, or workspace
         * leasing.
         */
        virtual oid linear_impl(const LinearRequest& request);
        /**
         * Backend hook behind `linear_workspace_requirements`. Receives the
         * already fully validated views, row window, layout mode, and head
         * scalars and must stay pure: no allocation, view snapshot or
         * request construction, registration, lease, token/queue resource,
         * submission, or retained view reference, and no dependence on free
         * arena capacity, fragmentation, queue occupancy, or completion
         * state. A backend that does not implement the operation throws
         * `UnsupportedOperation` here, and common validation always precedes
         * this capability decision.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                linear_workspace_requirements_impl(
                        const TensorView& x, const TensorView& w,
                        const TensorView& out, std::size_t s, std::size_t R,
                        LinearOutputLayout layout, std::size_t H,
                        std::size_t D);
        virtual oid rmsnorm_impl(const RmsnormRequest& request);
        /**
         * Immutable RMS normalization capability for one already validated
         * applicable leaf. A ported backend returns true exactly for the
         * leaves it queues; the default common implementation reports every
         * leaf unsupported, so a valid-shape request on an unported backend
         * reaches explicit capability rejection and stays `Unsupported`
         * before any workspace inspection, owner registration, sequence
         * consumption, or dispatch. Pure: it allocates nothing and has no
         * registration, lease, token/queue, or backend effect. It is only
         * ever asked about the nine applicable floating leaves; recognized
         * inapplicable leaves and non-`NONE` quantization are rejected by
         * common validation first.
         */
        [[nodiscard]] virtual bool rmsnorm_supported(
                DataType data_type) const;
        /**
         * Immutable RoPE execution hook. The default common implementation
         * reports `Unsupported`; a backend replaces it only after common
         * admission has completed and receives no borrowed caller view.
         */
        virtual oid rope_impl(const RopeRequest& request);
        /**
         * Pure capability and workspace hook for the RoPE requirement query.
         * A supported backend must report exactly `{0, 1}` without queue,
         * allocation, registration, lease, or native effects. The default
         * common implementation throws `UnsupportedOperation`.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                rope_workspace_requirements(const RopeRequest& request);

        virtual oid sdpa_impl(const SdpaRequest& request);
        /**
         * Pure SDPA capability/workspace hook. The request contains only
         * validated fixed-capacity metadata and checked derived dimensions;
         * no owner registration, lease, token, queue, or native effect may
         * occur here. Unported backends report UnsupportedOperation.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                sdpa_workspace_requirements_impl(
                        const SdpaRequest& request);
        /**
         * Backend hook behind `cache_append_workspace_requirements`.
         * Receives an already fully validated cache append request snapshot
         * and must stay pure: no allocation beyond the returned value, no
         * registration, lease, token/queue resource, or backend effect.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                cache_append_workspace_requirements(
                        const CacheAppendRequest& request);
        /**
         * Backend hook behind the four *_workspace_requirements queries.
         * Receives an already fully validated request snapshot and must
         * stay pure: no allocation beyond the returned value, no
         * registration, lease, token/queue resource, or backend effect.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                binary_workspace_requirements(const BinaryRequest& request);
        /**
         * Backend hook behind `embedding_workspace_requirements`. Receives
         * the already fully validated views and must stay pure: no
         * allocation, registration, lease, token/queue resource, submission,
         * index read, or retained view reference, and no dependence on free
         * arena capacity, fragmentation, queue occupancy, or completion
         * state. A backend that does not implement the operation reports
         * `UnsupportedOperation` here, and common validation always precedes
         * this capability decision.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                embedding_workspace_requirements_impl(
                        const TensorView& table, const TensorView& indices,
                        const TensorView& out);
        [[nodiscard]] const Device& queue_device() const;
        virtual void fence_through_sequence(
                std::uint64_t sequence) noexcept;
        void record_post_completion_failure(
                std::uint64_t sequence, std::exception_ptr failure);
        [[nodiscard]] static BinaryRequest validate_binary(
                const Device& device, BinaryOperation operation,
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] static BinaryViewSnapshot snapshot_binary_view(
                const TensorView& view,
                std::span<const std::size_t> result_dimensions);
        [[nodiscard]] static CacheAppendRequest validate_cache_append(
                const Device& device, const TensorView& source,
                const TensorView& destination, std::size_t a);
        [[nodiscard]] static CacheAppendViewSnapshot snapshot_cache_append_view(
                const TensorView& view);
        static void validate_copy(
                const Device& device, const TensorView& source,
                const TensorView& destination);
        [[nodiscard]] static LinearViewSnapshot snapshot_linear_view(
                const TensorView& view);
        [[nodiscard]] static RopeViewSnapshot snapshot_rope_view(
                const TensorView& view,
                const detail::CheckedViewFacts& facts);

        [[nodiscard]] static SiLUViewSnapshot snapshot_silu_view(
                const TensorView& view);
        [[nodiscard]] static SdpaViewSnapshot snapshot_sdpa_view(
                const TensorView& view,
                const detail::CheckedViewFacts& facts);
        [[nodiscard]] static SdpaRequest validate_sdpa(
                const Device& device, const TensorView& q,
                const TensorView& k, const TensorView& v,
                const TensorView& out, std::size_t a, std::size_t L);
        /**
         * Complete common linear projection admission validation in the
         * frozen contract order: recognized encodings, rank and nonzero
         * extents, the exact output shape and layout relation, matching
         * leading tuples, rank growth, the selected row window, then exact
         * device identity, live owners, stable handles, selected-plane
         * bounds and transformed strides with all checked arithmetic, then
         * conservative output/input overlap, and finally one applicable
         * leaf type with `QuantizationFormat::NONE` for all three views.
         * Allocation-free and effect-free: it inspects live metadata only
         * and builds no request or owned snapshot, so the pure requirement
         * query shares exactly these checks. The backend capability is
         * consulted afterwards by the operation's own hook.
         */
        static void validate_linear(
                const Device& device, const TensorView& x,
                const TensorView& w, const TensorView& out,
                std::size_t s, std::size_t R, LinearOutputLayout layout,
                std::size_t H, std::size_t D);
        static void validate_views(const Device& device,
                                   std::initializer_list<const TensorView*> views);
        [[nodiscard]] static bool identical_window(
                const TensorView& source, const TensorView& destination);
        [[nodiscard]] virtual std::string_view backend_label() const noexcept {
            return "unknown";
        }
        [[nodiscard]] static std::runtime_error unsupported(
                std::string_view backend, std::string_view operation);

        // Admission callbacks are prepared synchronously, then retained in
        // an immutable FIFO node until a native credit is available.
        oid submit_prepared(
                std::function<void(std::uint64_t)> prepare,
                std::function<void(std::uint64_t)> dispatch,
                std::function<void()> rollback = {});
        template <typename QueueWork>
        oid submit(QueueWork queue_work) {
            auto work = std::make_shared<QueueWork>(std::move(queue_work));
            return submit_prepared(
                    {}, [work](std::uint64_t sequence) {
                        (*work)(sequence);
                    });
        }
        // A fence factory produces the per-submission fence. Constant
        // fences (a single `Fence` reused for every submission) wrap the
        // existing constant in a factory that returns a copy of it; a
        // backend whose fence depends on the assigned sequence (e.g. to
        // carry the sequence through the registry entry for the parked
        // seam) builds a fresh fence inside the factory.
        using FenceFactory =
                std::function<detail::Fence(std::uint64_t sequence)>;

        template <typename QueueWork>
        oid submit_copy(
                const TensorView& source, const TensorView& destination,
                detail::RegistryState& state, detail::QueueId queue_id,
                FenceFactory build_fence, QueueWork queue_work) {
            struct Prepared {
                CopyRequest request;
                detail::EntryRegistration entries;
                std::function<void(
                        std::uint64_t, const CopyRequest&,
                        detail::EntryRegistration)> work;
                Prepared(
                        const TensorView& source_,
                        const TensorView& destination_, bool no_op_,
                        QueueWork work_)
                    : request(source_, destination_, no_op_),
                      work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    source, destination, identical_window(source, destination),
                    std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        prepared->entries = detail::register_copy_entries(
                                state, queue_id, sequence,
                                prepared->request.source.native_handle,
                                prepared->request.destination.native_handle,
                                build_fence(sequence));
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.source != 0) {
                            state.registry.remove_entry_if_present(
                                    prepared->entries.source,
                                    prepared->request.source.native_handle);
                        }
                        if (prepared->entries.destination != 0) {
                            state.registry.remove_entry_if_present(
                                    prepared->entries.destination,
                                    prepared->request.destination.native_handle);
                        }
                    });
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename QueueWork>
        oid submit_copy(
                const TensorView& source, const TensorView& destination,
                detail::RegistryState& state, detail::QueueId queue_id,
                const detail::Fence& fence, QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_copy(
                    source, destination, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }

        /**
         * Shared prepared-ownership submission for operations with exactly
         * three owner-registered views and an optional leased workspace.
         * `Request` is the immutable admission snapshot type; it must expose
         * `workspace`, `workspace_requirements`, and `workspace_lease`, as
         * `BinaryRequest` and `EmbeddingRequest` do. The retained copy is
         * registered in one all-or-nothing prepare step — workspace lease
         * first, then the three owner records through the existing registry
         * record types — before the FIFO node is published, and its rollback
         * releases partial registrations and leases. Binary and embedding
         * share this one ownership mechanism.
         */
        template <typename Request, typename QueueWork>
        oid submit_three_owner_request(
                const Request& request,
                const std::array<detail::BinaryOwnerRegistration, 3>& owners,
                detail::RegistryState& state, detail::QueueId queue_id,
                FenceFactory build_fence, QueueWork queue_work) {
            struct Prepared {
                Request request;
                detail::WorkspaceLease workspace_lease;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const Request&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(
                        const Request& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, owners, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        try {
                            if (prepared->request.workspace_requirements.bytes
                                    != 0) {
                                prepared->workspace_lease =
                                        detail::acquire_workspace_lease(
                                                state,
                                                prepared->request.workspace
                                                        .owner_identity(),
                                                detail::WorkspaceValidation::range(
                                                        prepared->request.workspace),
                                                sequence, queue_id, fence);
                                prepared->request.workspace_lease =
                                        prepared->workspace_lease;
                            }
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            if (prepared->workspace_lease.entry_id != 0) {
                                detail::complete_workspace_lease(
                                        state, prepared->workspace_lease.entry_id, true);
                                prepared->workspace_lease = {};
                                prepared->request.workspace_lease = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                        if (prepared->workspace_lease.entry_id != 0) {
                            detail::complete_workspace_lease(
                                    state, prepared->workspace_lease.entry_id, true);
                        }
                    });
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename Request, typename QueueWork>
        oid submit_three_owner_request(
                const Request& request,
                const std::array<detail::BinaryOwnerRegistration, 3>& owners,
                detail::RegistryState& state, detail::QueueId queue_id,
                const detail::Fence& fence, QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_three_owner_request(
                    request, owners, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }
        /**
         * Narrow prepared-ownership adapter for SDPA. It intentionally keeps
         * its own fixed four-owner registration shape: Q/K/V read aliases
         * deduplicate by identity, while the output is independently
         * validated as disjoint by common admission.
         */
        template <typename QueueWork>
        oid submit_sdpa(
                const SdpaRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            struct Prepared {
                SdpaRequest request;
                detail::WorkspaceLease workspace_lease;
                detail::SdpaEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const SdpaRequest&,
                        detail::SdpaEntryRegistration)> work;
                Prepared(const SdpaRequest& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        const std::array<detail::SdpaOwnerRegistration, 4>
                                owners{{
                                        {prepared->request.q.owner_identity,
                                         prepared->request.q.native_handle},
                                        {prepared->request.k.owner_identity,
                                         prepared->request.k.native_handle},
                                        {prepared->request.v.owner_identity,
                                         prepared->request.v.native_handle},
                                        {prepared->request.out.owner_identity,
                                         prepared->request.out.native_handle},
                                }};
                        try {
                            if (prepared->request.workspace_requirements.bytes
                                    != 0) {
                                prepared->workspace_lease =
                                        detail::acquire_workspace_lease(
                                                state,
                                                prepared->request.workspace
                                                        .owner_identity(),
                                                detail::WorkspaceValidation::range(
                                                        prepared->request.workspace),
                                                sequence, queue_id, fence);
                                prepared->request.workspace_lease =
                                        prepared->workspace_lease;
                            }
                            prepared->entries = detail::register_sdpa_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            if (prepared->workspace_lease.entry_id != 0) {
                                detail::complete_workspace_lease(
                                        state, prepared->workspace_lease.entry_id, true);
                                prepared->workspace_lease = {};
                                prepared->request.workspace_lease = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                        if (prepared->workspace_lease.entry_id != 0) {
                            detail::complete_workspace_lease(
                                    state, prepared->workspace_lease.entry_id, true);
                        }
                    });
        }

        template <typename QueueWork>
        oid submit_sdpa(
                const SdpaRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_sdpa(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }


        // Admission path used by a ported SiLU hook. The fixed-size request
        // is captured first, then both distinct owners are registered
        // transactionally through the existing FIFO prepare/dispatch/rollback
        // machinery until proven completion. SiLU consumes no raw workspace.
        template <typename QueueWork>
        oid submit_silu(
                const SiLURequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            struct Prepared {
                SiLURequest request;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const SiLURequest&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(const SiLURequest& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        const std::array<detail::BinaryOwnerRegistration, 2>
                                owners{{
                                        {prepared->request.x.owner_identity,
                                         prepared->request.x.native_handle},
                                        {prepared->request.y.owner_identity,
                                         prepared->request.y.native_handle},
                                }};
                        try {
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                    });
        }

        template <typename QueueWork>
        oid submit_silu(
                const SiLURequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_silu(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }
        // Admission path used by a ported RMS normalization hook. The
        // immutable request is captured first, then every distinct owner is
        // registered through the existing in-order prepare/register/dispatch/
        // rollback machinery until proven completion. RMS normalization
        // consumes no raw workspace, so no lease is acquired and only the
        // read/read `x`/`scale` identities are deduplicated.

        template <typename QueueWork>
        oid submit_rmsnorm(
                const RmsnormRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            struct Prepared {
                RmsnormRequest request;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const RmsnormRequest&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(const RmsnormRequest& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        const std::array<detail::BinaryOwnerRegistration, 3>
                                owners{{
                                        {prepared->request.x.owner_identity,
                                         prepared->request.x.native_handle},
                                        {prepared->request.scale.owner_identity,
                                         prepared->request.scale.native_handle},
                                        {prepared->request.out.owner_identity,
                                         prepared->request.out.native_handle},
                                }};
                        try {
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                    });
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename QueueWork>
        oid submit_rmsnorm(
                const RmsnormRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_rmsnorm(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }
        /**
         * Prepared ownership adapter for one RoPE request. The two admitted
         * owners and any future positive workspace lease are registered as
         * one all-or-nothing prepare step and released on rollback or proven
         * completion. The output owner is distinct by common admission, so
         * no alias category is introduced here.
         */
        template <typename QueueWork>
        oid submit_rope(
                const RopeRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            struct Prepared {
                RopeRequest request;
                detail::WorkspaceLease workspace_lease;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const RopeRequest&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(const RopeRequest& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        const std::array<detail::BinaryOwnerRegistration, 2>
                                owners{{
                                        {prepared->request.x.owner_identity,
                                         prepared->request.x.native_handle},
                                        {prepared->request.out.owner_identity,
                                         prepared->request.out.native_handle},
                                }};
                        try {
                            if (prepared->request.workspace_requirements.bytes
                                    != 0) {
                                prepared->workspace_lease =
                                        detail::acquire_workspace_lease(
                                                state,
                                                prepared->request.workspace
                                                        .owner_identity(),
                                                detail::WorkspaceValidation::range(
                                                        prepared->request.workspace),
                                                sequence, queue_id, fence);
                                prepared->request.workspace_lease =
                                        prepared->workspace_lease;
                            }
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            if (prepared->workspace_lease.entry_id != 0) {
                                detail::complete_workspace_lease(
                                        state, prepared->workspace_lease.entry_id, true);
                                prepared->workspace_lease = {};
                                prepared->request.workspace_lease = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                        if (prepared->workspace_lease.entry_id != 0) {
                            detail::complete_workspace_lease(
                                    state, prepared->workspace_lease.entry_id, true);
                        }
                    });
        }

        // Constant-fence overload retained so backend queues can use one
        // fence object when their native submission path already provides it.
        template <typename QueueWork>
        oid submit_rope(
                const RopeRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_rope(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }


        template <typename QueueWork>
        oid submit_binary(
                const BinaryRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            return submit_three_owner_request(
                    request,
                    std::array<detail::BinaryOwnerRegistration, 3>{{
                            {request.lhs.owner_identity,
                             request.lhs.native_handle},
                            {request.rhs.owner_identity,
                             request.rhs.native_handle},
                            {request.out.owner_identity,
                             request.out.native_handle}}},
                    state, queue_id, build_fence, std::move(queue_work));
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename QueueWork>
        oid submit_binary(
                const BinaryRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_binary(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }
        /**
         * Shared prepared-ownership submission for cache row append. The
         * source and destination owners are retained through completion and
         * any positive caller workspace is leased before backend work is
         * invoked. Registration and lease acquisition are all-or-nothing;
         * every partial resource is removed by rollback.
         */
        template <typename QueueWork>
        oid submit_cache_append(
                const CacheAppendRequest& request,
                detail::RegistryState& state, detail::QueueId queue_id,
                FenceFactory build_fence, QueueWork queue_work) {
            struct Prepared {
                CacheAppendRequest request;
                detail::WorkspaceLease workspace_lease;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const CacheAppendRequest&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(
                        const CacheAppendRequest& request_,
                        QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, build_fence](
                            std::uint64_t sequence) {
                        const detail::Fence fence = build_fence(sequence);
                        const std::array<
                                detail::BinaryOwnerRegistration, 2>
                                owners{{
                                        {prepared->request.source.owner_identity,
                                         prepared->request.source.native_handle},
                                        {prepared->request.destination.owner_identity,
                                         prepared->request.destination.native_handle},
                                }};
                        try {
                            if (prepared->request.workspace_requirements.bytes
                                    != 0) {
                                prepared->workspace_lease =
                                        detail::acquire_workspace_lease(
                                                state,
                                                prepared->request.workspace
                                                        .owner_identity(),
                                                detail::WorkspaceValidation::range(
                                                        prepared->request.workspace),
                                                sequence, queue_id, fence);
                                prepared->request.workspace_lease =
                                        prepared->workspace_lease;
                            }
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            if (prepared->workspace_lease.entry_id != 0) {
                                detail::complete_workspace_lease(
                                        state, prepared->workspace_lease.entry_id, true);
                                prepared->workspace_lease = {};
                                prepared->request.workspace_lease = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                        if (prepared->workspace_lease.entry_id != 0) {
                            detail::complete_workspace_lease(
                                    state, prepared->workspace_lease.entry_id, true);
                        }
                    });
        }

        // Constant-fence overload retained so backend queue implementations
        // that build one fence per operation keep the same admission seam.
        template <typename QueueWork>
        oid submit_cache_append(
                const CacheAppendRequest& request,
                detail::RegistryState& state, detail::QueueId queue_id,
                const detail::Fence& fence, QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_cache_append(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }

        template <typename QueueWork>
        oid submit_embedding(
                const EmbeddingRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            return submit_three_owner_request(
                    request,
                    std::array<detail::BinaryOwnerRegistration, 3>{{
                            {request.table.owner_identity,
                             request.table.native_handle},
                            {request.indices.owner_identity,
                             request.indices.native_handle},
                            {request.out.owner_identity,
                             request.out.native_handle}}},
                    state, queue_id, build_fence, std::move(queue_work));
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename QueueWork>
        oid submit_embedding(
                const EmbeddingRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_embedding(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }

        /**
         * Shared prepared-ownership submission for one admitted linear
         * projection: the distinct `x`, `w`, and `out` owners — with
         * permitted read/read aliases deduplicated by the registration
         * pass — and the optional leased workspace are retained through
         * proven completion by the same all-or-nothing
         * prepare/register/dispatch/rollback mechanism the binary and
         * embedding paths use.
         */
        template <typename QueueWork>
        oid submit_linear(
                const LinearRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, FenceFactory build_fence,
                QueueWork queue_work) {
            return submit_three_owner_request(
                    request,
                    std::array<detail::BinaryOwnerRegistration, 3>{{
                            {request.x.owner_identity,
                             request.x.native_handle},
                            {request.w.owner_identity,
                             request.w.native_handle},
                            {request.out.owner_identity,
                             request.out.native_handle}}},
                    state, queue_id, build_fence, std::move(queue_work));
        }

        // Constant-fence overload retained so existing callers that build
        // the fence once per dispatch keep compiling unchanged.
        template <typename QueueWork>
        oid submit_linear(
                const LinearRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            const detail::Fence fence_copy = fence;
            return submit_linear(
                    request, state, queue_id,
                    [fence_copy](std::uint64_t) { return fence_copy; },
                    std::move(queue_work));
        }

        void complete(std::uint64_t sequence,
                      std::exception_ptr failure = nullptr);
        void commit_failure(std::uint64_t sequence,
                            std::exception_ptr failure);
        void seek_next_sequence(std::uint64_t next_sequence);
        // Backend destructors call this before their worker/native members
        // disappear. It closes acceptance and drains accepted FIFO nodes.
        void close_and_drain() noexcept;

    private:
        [[nodiscard]] oid submit_binary_operation(
                BinaryOperation operation, const TensorView& lhs,
                const TensorView& rhs, TensorView& out,
                RawWorkspaceView workspace) noexcept;
        [[nodiscard]] oid invoke(oid result) noexcept;
        [[nodiscard]] oid invoke_failure(std::exception_ptr failure) noexcept;
        [[nodiscard]] static oid map_failure(
                std::exception_ptr failure) noexcept;
        [[nodiscard]] static std::uint8_t lease_queue_id();
        static void release_queue_id(std::uint8_t queue_id) noexcept;
        [[nodiscard]] oid encode_token(std::uint64_t sequence) const noexcept;
        const Device* device_ = nullptr;
        std::uint8_t queue_id_ = 0;
        static constexpr std::uint64_t kSequenceBits = 55;
        static constexpr std::uint64_t kSequenceMask =
                (std::uint64_t{1} << kSequenceBits) - 1;
        static constexpr std::uint64_t kMaxSequence = kSequenceMask;
        std::map<std::uint64_t, std::uint64_t> skipped_sequences_;
        std::uint64_t next_sequence_ = 1;
        std::uint64_t completed_ = 0;
        std::mutex completion_mutex_;
        std::condition_variable completion_cv_;
        std::map<std::uint64_t, std::exception_ptr> failures_;
        std::map<std::uint64_t, std::exception_ptr> pending_failures_;
        struct AdmissionNode {
            std::function<void(std::uint64_t)> dispatch;
            // Resources prepared before admission are released if queue
            // teardown drains this parked node before dispatch.
            std::function<void()> rollback;
            bool executing = false;
        };
        void pump_admission(
                std::exception_ptr* synchronous_failure = nullptr,
                std::uint64_t synchronous_sequence = 0) noexcept;
        void abandon_sequence(std::uint64_t sequence) noexcept;
        std::map<std::uint64_t, AdmissionNode> admission_nodes_;
        std::deque<std::uint64_t> admission_fifo_;
        std::size_t admission_credits_ = 0;
        std::size_t admission_capacity_ = 16;
        bool admission_pumping_ = false;
        bool admission_closing_ = false;
        bool queue_slot_reserved_ = false;
    };

    class Block {
    public:
        virtual ~Block() = default;
        virtual void forward(const Tensor& x, Tensor& y) = 0;
    };
}
