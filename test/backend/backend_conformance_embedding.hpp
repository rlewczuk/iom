#pragma once

// Backend-neutral embedding-lookup ("gather") conformance: the independent
// raw-code row-selection reference, the explicit per-driver capability
// declaration, and every shared observable case of the operation-owned
// Embedding lookup contract in `docs/BACKEND_CONTRACT.md`.
//
// The reference encodes the table as row-major raw bytes with the existing
// test-only LSB-first bit encoder and selects whole rows through the index
// equation `out[b, r, f] = table[indices[b, 0, r], f]`. It never calls
// production addressing, codec, gather, or CPU embedding helpers; the only
// reused test-only machinery is the raw bit reader/writer and the independent
// canonical 16x16 storage mapping. Comparisons are raw-bit exact with zero
// numeric tolerance.
//
// The harness never switches on BackendKind. Each driver supplies an explicit
// EmbeddingDeclaration naming the payload and index leaves its port
// implements, the matrix that port must reach, and its exact workspace
// requirement. An empty implemented span declares an unported port: the suite
// then observes capability rejection only and never claims gather conformance,
// because a capability rejection is not numerical success.

#include "backend/backend_conformance_oracle.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Published matrices and the explicit per-driver declaration.
// ---------------------------------------------------------------------------

// The complete standard matrix every CPU, CUDA, ROCm, and SYCL port must
// reach: all 23 payload leaves and all 12 integral index leaves.
inline constexpr std::array<iom::DataType, 23> kEmbeddingPayloadSpan = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F8_E8M0,
        iom::DataType::F16, iom::DataType::BF16,
        iom::DataType::F32, iom::DataType::F64,
};

inline constexpr std::array<iom::DataType, 12> kEmbeddingIdSpan = {
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
};

// The explicitly temporary one-carrier TTNN expectation: 19 payload leaves
// (all except `F8_E8M0`, `I64`, `U64`, and `F64`) and the 10 narrow index
// leaves (all except `I64` and `U64`). The wide-carrier leaf replaces both
// with the final TTNN 22/12 matrix, so a staged span is a temporary
// intermediate declaration and never final support.
inline constexpr std::array<iom::DataType, 19>
        kEmbeddingStagedTtnnPayloadSpan = {
                iom::DataType::BOOL,
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
                iom::DataType::F4_E2M1,
                iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
                iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
                iom::DataType::F16, iom::DataType::BF16,
                iom::DataType::F32,
        };

inline constexpr std::array<iom::DataType, 10>
        kEmbeddingStagedTtnnIdSpan = {
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
        };

// `BOOL` and the ten floating leaves are recognized but inapplicable as index
// semantics: the operation capability stage reports them as `Unsupported`.
inline constexpr std::array<iom::DataType, 11> kEmbeddingNonIdSpan = {
        iom::DataType::BOOL,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F8_E8M0,
        iom::DataType::F16, iom::DataType::BF16,
        iom::DataType::F32, iom::DataType::F64,
};

// The explicit empty span of an unported port.
inline constexpr std::span<const iom::DataType> kNoEmbeddingSpan{};

// One driver's explicit embedding declaration.
struct EmbeddingDeclaration {
    // The complete payload and index matrix this port is expected to reach.
    // The TTNN driver passes the staged matrix here and sets `staged`.
    std::span<const iom::DataType> target_payloads;
    std::span<const iom::DataType> target_ids;
    // The leaves this revision's backend port implements and the suite
    // therefore observes numerically. Empty means unported: the suite asserts
    // capability rejection only and never reports gather success.
    std::span<const iom::DataType> implemented_payloads;
    std::span<const iom::DataType> implemented_ids;
    // True exactly while `target_payloads`/`target_ids` are an explicitly
    // temporary staged matrix that the port's next leaf must replace with the
    // final matrix. It never advertises support.
    bool staged = false;
    // The exact workspace requirement the port reports for an applicable
    // request: `{0, 1}` on CPU and `{32, 32}` on CUDA, ROCm, SYCL, and TTNN.
    // A positive range is an exact live device owner whose first status word
    // is `0` valid / `1` invalid ID and whose remaining 28 bytes are reserved
    // control padding that the caller neither initializes nor polls.
    iom::WorkspaceRequirements workspace{};
};

[[nodiscard]] inline bool embedding_declares(
        std::span<const iom::DataType> leaves, iom::DataType value) {
    return std::find(leaves.begin(), leaves.end(), value) != leaves.end();
}

// A request is implemented only when both its payload and its index leaf are
// declared. Every other combination must stay `Unsupported`: the declared
// capability stage already rejects inapplicable index leaves, and a port that
// does not implement a leaf must reject it rather than silently accept it.
[[nodiscard]] inline bool embedding_implements(
        const EmbeddingDeclaration& declaration, iom::DataType payload,
        iom::DataType id_type) {
    return embedding_declares(declaration.implemented_payloads, payload)
           && embedding_declares(declaration.implemented_ids, id_type);
}

[[nodiscard]] inline iom::TensorSpec embedding_spec(
        std::vector<std::size_t> dimensions, iom::DataType data_type) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, data_type};
}

// The leaves of `probe` that the candidate can actually store, which is the
// only set a fixture may instantiate. The device's own capability table is the
// existing, backend-neutral answer; a fixture never assumes a leaf the device
// would refuse to allocate.
[[nodiscard]] inline std::vector<iom::DataType> embedding_storable(
        std::span<const iom::DataType> probe,
        std::span<const iom::DataType> supported) {
    std::vector<iom::DataType> leaves;
    for (const iom::DataType leaf : probe) {
        if (embedding_declares(supported, leaf)) {
            leaves.push_back(leaf);
        }
    }
    return leaves;
}

// ---------------------------------------------------------------------------
// Independent reference: raw row selection on separately encoded bytes.
// ---------------------------------------------------------------------------

// Deliberate negative variants. They exist only so the self-check can prove
// the reference detects a wrong row, a transposed table axis, a padded table
// stride, a numeric re-encoding, and a plane-blind leading block, and that
// this detection is selective rather than unconditional.
enum class EmbeddingOracleVariant {
    honest,
    wrong_row,
    transposed,
    padded_table,
    numeric_reencode,
    plane_blind,
};

// Raw reader over one fixture image. A read that would cross the image returns
// zero, which is exactly how an oracle that addresses padded storage beyond
// the logical table behaves.
[[nodiscard]] inline std::uint64_t read_bits_clamped(
        std::span<const std::byte> image, std::size_t bit_offset,
        std::size_t nbits) {
    if (bit_offset > image.size() * 8
            || nbits > image.size() * 8 - bit_offset) {
        return 0;
    }
    return read_storage_bits(image.data(), bit_offset, nbits);
}

// Raw code of one logical row-major element of one fixture image.
[[nodiscard]] inline std::uint64_t embedding_image_code(
        std::span<const std::byte> image, iom::DataType type,
        std::size_t linear) {
    const std::size_t bits = bits_of(type);
    return read_bits_clamped(image, linear * bits, bits);
}

// A codec that converts instead of copying: a subnormal magnitude is flushed
// to signed zero and any NaN payload is canonicalized. The gather contract
// copies raw bits, so this variant exists only to prove the reference detects
// a numeric re-encoding of signed zero, subnormals, and NaN payloads. Only the
// 16-bit floating payload is modelled; every other payload is unchanged.
[[nodiscard]] inline std::uint64_t embedding_recode_numeric(
        iom::DataType payload, std::uint64_t code) {
    if (payload != iom::DataType::BF16) {
        return code;
    }
    const std::uint16_t raw = static_cast<std::uint16_t>(code);
    if ((raw & 0x7F80u) == 0) {
        return static_cast<std::uint16_t>(raw & 0x8000u);
    }
    if ((raw & 0x7F80u) == 0x7F80u && (raw & 0x007Fu) != 0) {
        return 0x7FC0u;
    }
    return raw;
}

// The independent reference. `table_row_major` holds the table's own
// row-major logical bytes at `bits_of(payload)` bits per element, and `codes`
// holds one integral index code per output row, planes consecutive and rows
// consecutive: `codes[plane * run + row]` selects the row of that output row.
// Whole rows are copied bit-exactly with no arithmetic, re-encoding, or
// tolerance.
[[nodiscard]] inline std::vector<std::byte> select_embedding_rows(
        std::span<const std::byte> table_row_major, iom::DataType payload,
        std::size_t vocabulary, std::size_t features, std::size_t run,
        std::span<const std::uint64_t> row_codes,
        EmbeddingOracleVariant variant = EmbeddingOracleVariant::honest) {
    const std::size_t bits = bits_of(payload);
    REQUIRE(vocabulary != 0);
    REQUIRE(features != 0);
    REQUIRE(run != 0);
    REQUIRE(row_codes.size() % run == 0);
    const std::size_t planes = row_codes.size() / run;
    const std::size_t elements = planes * run * features;
    std::vector<std::byte> selected((elements * bits + 7) / 8, std::byte{0});

    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < run; ++row) {
            std::size_t code = static_cast<std::size_t>(
                    row_codes[plane * run + row]);
            if (variant == EmbeddingOracleVariant::wrong_row) {
                code = (code + 1) % vocabulary;
            }
            if (variant == EmbeddingOracleVariant::plane_blind) {
                code = static_cast<std::size_t>(row_codes[row]);
            }
            REQUIRE(code < vocabulary);
            for (std::size_t feature = 0; feature < features; ++feature) {
                std::size_t source = code * features + feature;
                if (variant == EmbeddingOracleVariant::transposed) {
                    source = feature * vocabulary + code;
                }
                if (variant == EmbeddingOracleVariant::padded_table) {
                    source = code * canonical_padded_extent(features)
                           + feature;
                }
                std::uint64_t value =
                        read_bits_clamped(table_row_major, source * bits, bits);
                if (variant == EmbeddingOracleVariant::numeric_reencode) {
                    value = embedding_recode_numeric(payload, value);
                }
                write_storage_bits(
                        selected.data(),
                        ((plane * run + row) * features + feature) * bits, bits,
                        value);
            }
        }
    }
    return selected;
}

// Deterministic exact index codes of one valid fixture: the first and the last
// vocabulary row, a repeat of each, and salted interior codes, all inside
// `[0, V)` and therefore representable in the leaf's nonnegative domain
// without narrowing.
[[nodiscard]] inline std::uint64_t embedding_valid_code(
        std::size_t vocabulary, std::size_t run, std::size_t plane,
        std::size_t row, std::uint64_t salt) {
    REQUIRE(vocabulary != 0);
    if (row == 0 || run == 1) {
        return 0;
    }
    if (row == 1 || row == 2) {
        return vocabulary - 1;
    }
    if (row == 3) {
        return 0;
    }
    return splitmix64(splitmix64(salt + plane) + row) % vocabulary;
}

// One integral index leaf's own domain. `nonnegative_max` is the largest code
// that leaf represents as a nonnegative value, so every fixture code stays
// representable in the leaf's width.
struct EmbeddingIdDomain {
    std::size_t bits = 0;
    bool is_signed = false;
    std::uint64_t nonnegative_max = 0;
};

[[nodiscard]] inline EmbeddingIdDomain embedding_id_domain(
        iom::DataType type) {
    const std::size_t bits = bits_of(type);
    switch (type) {
        case iom::DataType::I2:
        case iom::DataType::I4:
        case iom::DataType::I8:
        case iom::DataType::I16:
        case iom::DataType::I32:
        case iom::DataType::I64:
            return {bits, true, (std::uint64_t{1} << (bits - 1)) - 1};
        case iom::DataType::U2:
        case iom::DataType::U4:
        case iom::DataType::U8:
        case iom::DataType::U16:
        case iom::DataType::U32:
        case iom::DataType::U64:
            return {bits, false,
                    bits >= 64 ? std::numeric_limits<std::uint64_t>::max()
                               : (std::uint64_t{1} << bits) - 1};
        default:
            break;
    }
    REQUIRE_MESSAGE(false, "embedding_id_domain: not an integral index leaf");
    return {};
}

// Vocabulary of a fixture for one index leaf. Small widths stay inside their
// own nonnegative domain so the positive out-of-vocabulary code `V` is itself
// representable without narrowing.
[[nodiscard]] inline std::size_t embedding_vocabulary_for(
        iom::DataType id_type, std::size_t requested) {
    const EmbeddingIdDomain domain = embedding_id_domain(id_type);
    const std::uint64_t bounded = std::min<std::uint64_t>(
            domain.nonnegative_max, static_cast<std::uint64_t>(requested));
    REQUIRE(bounded != 0);
    return static_cast<std::size_t>(bounded);
}

// The out-of-vocabulary and negative code classes of one index leaf, every one
// representable in that leaf's width. `positive` is `>= V`, `maximum` is the
// leaf's largest code (`U64_MAX` on `U64`), and `negative` is the sign-bit
// pattern of a signed leaf; `has_negative` is false for unsigned leaves, which
// have no negative code.
struct EmbeddingOovCodes {
    std::uint64_t positive = 0;
    std::uint64_t maximum = 0;
    std::uint64_t negative = 0;
    bool has_negative = false;
};

[[nodiscard]] inline EmbeddingOovCodes embedding_oov_codes(
        iom::DataType id_type, std::size_t vocabulary) {
    const EmbeddingIdDomain domain = embedding_id_domain(id_type);
    REQUIRE(vocabulary <= domain.nonnegative_max);
    EmbeddingOovCodes codes;
    codes.positive = vocabulary;
    codes.maximum = domain.nonnegative_max;
    codes.has_negative = domain.is_signed;
    if (domain.is_signed) {
        codes.negative = std::uint64_t{1} << (domain.bits - 1);
    }
    return codes;
}

// ---------------------------------------------------------------------------
// Request fixtures: owner specs, encoded index bytes, and poison images.
// ---------------------------------------------------------------------------

// One shared embedding request case: exact runtime extents, the table owner's
// leading plane count, the index/output owner leading tuple, and the
// leading-only transform applied identically to the index and output owners
// (so both keep exactly the same leading tuple while each uses its own
// independent owner offset and strides).
struct EmbeddingCase {
    std::string label;
    iom::DataType payload = iom::DataType::BF16;
    iom::DataType id_type = iom::DataType::U32;
    std::size_t vocabulary = 17;
    std::size_t features = 17;
    std::size_t run = 17;
    // Table owner leading planes: one keeps the owner rank two, and a larger
    // count makes the rank-two table view a selected plane of a larger owner.
    std::size_t table_planes = 1;
    std::vector<std::size_t> leading_dims;
    std::function<iom::TensorView(const iom::TensorView&)> transform;
    // Exact index code of one index owner plane and row. Omitted for the valid
    // fixtures, which use `embedding_valid_code`.
    std::function<std::uint64_t(std::size_t, std::size_t)> code;
    // True for the accepted out-of-vocabulary and negative-ID fixtures: the
    // output is unspecified and is never compared.
    bool out_of_vocabulary = false;
    // True when the table owner is seeded through the storage oracle with
    // non-logical padding forced nonzero.
    bool poison_native_padding = false;
    std::uint64_t salt = 0;
};

struct EmbeddingCaseSpecs {
    iom::TensorSpec table_owner;
    iom::TensorSpec index_owner;
    iom::TensorSpec out_owner;
};

[[nodiscard]] inline EmbeddingCaseSpecs embedding_case_specs(
        const EmbeddingCase& item) {
    std::vector<std::size_t> table_dims;
    if (item.table_planes > 1) {
        table_dims.push_back(item.table_planes);
    }
    table_dims.push_back(item.vocabulary);
    table_dims.push_back(item.features);
    std::vector<std::size_t> index_dims = item.leading_dims;
    index_dims.push_back(1);
    index_dims.push_back(item.run);
    std::vector<std::size_t> out_dims = item.leading_dims;
    out_dims.push_back(item.run);
    out_dims.push_back(item.features);
    return EmbeddingCaseSpecs{
            embedding_spec(std::move(table_dims), item.payload),
            embedding_spec(std::move(index_dims), item.id_type),
            embedding_spec(std::move(out_dims), item.payload)};
}

// One leading-only view transform, applied identically to both the index and
// the output owner so the two views keep the same leading tuple.
struct EmbeddingTransform {
    const char* label;
    std::function<iom::TensorView(const iom::TensorView&)> build;
};

[[nodiscard]] inline std::vector<EmbeddingTransform> embedding_transforms() {
    std::vector<EmbeddingTransform> transforms;
    transforms.push_back(
            {"full", [](const iom::TensorView& full) { return full; }});
    transforms.push_back(
            {"select of the first leading axis at its last index",
             [](const iom::TensorView& full) {
                 return full.select(
                         0, full.spec().shape.dimensions()[0] - 1);
             }});
    transforms.push_back(
            {"stepped slice of the first leading axis",
             [](const iom::TensorView& full) {
                 const std::size_t extent =
                         full.spec().shape.dimensions()[0];
                 return full.slice(0, 0, (extent + 1) / 2, 2);
             }});
    transforms.push_back(
            {"reversed leading permutation",
             [](const iom::TensorView& full) {
                 const std::size_t leading =
                         full.spec().shape.rank() - 2;
                 std::vector<std::size_t> order(leading);
                 for (std::size_t axis = 0; axis < leading; ++axis) {
                     order[axis] = leading - 1 - axis;
                 }
                 return full.permute(std::span<const std::size_t>{order});
             }});
    transforms.push_back(
            {"single leading axis reshape",
             [](const iom::TensorView& full) {
                 const std::span<const std::size_t> dims =
                         full.spec().shape.dimensions();
                 std::size_t planes = 1;
                 for (std::size_t axis = 0; axis + 2 < dims.size(); ++axis) {
                     planes *= dims[axis];
                 }
                 return full.reshape_leading(span_of({planes}));
             }});
    transforms.push_back(
            {"select, stepped slice, and permutation",
             [](const iom::TensorView& full) {
                 const std::size_t leading =
                         full.spec().shape.rank() - 2;
                 const std::size_t second =
                         full.spec().shape.dimensions()[1];
                 std::vector<std::size_t> order(leading - 1);
                 for (std::size_t axis = 0; axis + 1 < leading; ++axis) {
                     order[axis] = leading - 2 - axis;
                 }
                 return full.select(
                                0, full.spec().shape.dimensions()[0] - 1)
                         .slice(0, 0, (second + 1) / 2, 2)
                         .permute(std::span<const std::size_t>{order});
             }});
    return transforms;
}

// Exact index owner bytes: row-major over the owner's leading planes with the
// single-row axis and the run axis last, least-significant bit first, which is
// the raw transfer encoding every fixture and backend shares.
[[nodiscard]] inline std::vector<std::byte> encode_index_owner(
        const iom::TensorSpec& owner_spec, std::size_t run,
        const std::function<std::uint64_t(std::size_t, std::size_t)>& code) {
    const std::size_t bits = bits_of(owner_spec.data_type);
    REQUIRE(run != 0);
    const std::size_t planes = owner_spec.shape.element_count() / run;
    std::vector<std::byte> bytes(owner_spec.logical_nbytes(), std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(bytes.data());
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < run; ++row) {
            write_bits(
                    base, (plane * run + row) * bits, bits, code(plane, row));
        }
    }
    return bytes;
}

// Owner plane of one logical plane of a leading-only view, from the public
// view metadata alone: the selected plane offset plus the view's own leading
// strides over its own leading dimensions.
[[nodiscard]] inline std::size_t embedding_view_plane(
        const iom::TensorView& view, std::size_t logical_plane) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    const std::size_t leading = dims.size() - 2;
    const std::span<const std::size_t> strides = view.plane_strides();
    REQUIRE_EQ(strides.size(), leading);
    std::size_t plane = view.plane_offset();
    std::size_t rest = logical_plane;
    for (std::size_t axis = leading; axis-- > 0;) {
        plane += (rest % dims[axis]) * strides[axis];
        rest /= dims[axis];
    }
    return plane;
}

[[nodiscard]] inline std::size_t embedding_view_planes(
        const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis + 2 < dims.size(); ++axis) {
        planes *= dims[axis];
    }
    return planes;
}

// Elements of one logical plane, which every leading-only transform leaves
// unchanged.
[[nodiscard]] inline std::size_t embedding_plane_elements(
        const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    REQUIRE(dims.size() >= 2);
    return dims[dims.size() - 2] * dims[dims.size() - 1];
}

// The logical image of one leading-only view over one owner image: the view's
// own plane offset and strides select the owner plane and the final two
// coordinates stay unchanged.
[[nodiscard]] inline std::vector<std::byte> embedding_view_image(
        const iom::TensorView& view, std::span<const std::byte> owner_image) {
    const std::size_t bits = bits_of(view.spec().data_type);
    const std::size_t planes = embedding_view_planes(view);
    const std::size_t plane_elements = embedding_plane_elements(view);
    REQUIRE_EQ(owner_image.size(), view.owner_identity()->view().spec()
                                          .logical_nbytes());
    std::vector<std::byte> image(view.spec().logical_nbytes(), std::byte{0});
    for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::size_t owner_plane = embedding_view_plane(view, plane);
        for (std::size_t element = 0; element < plane_elements; ++element) {
            const std::uint64_t value = embedding_image_code(
                    owner_image, view.spec().data_type,
                    owner_plane * plane_elements + element);
            write_storage_bits(
                    image.data(), (plane * plane_elements + element) * bits,
                    bits, value);
        }
    }
    return image;
}

// Owner-level logical image of the table: the selected rank-two table view
// carries its own bytes and every other owner plane carries poison, so a
// gather that ignores the table view's plane selection reads poison instead of
// a valid row. Element-wise, so sub-byte payloads stay exact.
[[nodiscard]] inline std::vector<std::byte> table_owner_image(
        const iom::TensorSpec& owner_spec,
        const iom::TensorView& table_view,
        std::span<const std::byte> table_view_bytes,
        std::span<const std::byte> poison_bytes) {
    const std::size_t bits = bits_of(owner_spec.data_type);
    const std::size_t elements = owner_spec.shape.element_count();
    REQUIRE_EQ(poison_bytes.size(), owner_spec.logical_nbytes());
    REQUIRE_EQ(table_view.spec().shape.rank(), std::size_t{2});
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis + 2 < dims.size(); ++axis) {
        planes *= dims[axis];
    }
    const std::size_t plane_elements = elements / planes;
    const bool selects_plane = planes > 1;
    const std::size_t selected =
            selects_plane ? embedding_view_plane(table_view, 0) : 0;
    if (selects_plane) {
        REQUIRE_LT(selected, planes);
        REQUIRE_EQ(table_view_bytes.size(),
                   (plane_elements * bits + 7) / 8);
    } else {
        REQUIRE_EQ(table_view_bytes.size(), owner_spec.logical_nbytes());
    }
    std::vector<std::byte> image(owner_spec.logical_nbytes(), std::byte{0});
    for (std::size_t linear = 0; linear < elements; ++linear) {
        const bool inside = selects_plane && linear / plane_elements == selected;
        const std::uint64_t value = inside
                ? embedding_image_code(
                          table_view_bytes, owner_spec.data_type,
                          linear % plane_elements)
                : embedding_image_code(
                          selects_plane ? poison_bytes : table_view_bytes,
                          owner_spec.data_type, linear);
        write_storage_bits(image.data(), linear * bits, bits, value);
    }
    return image;
}

// Standard tiled owner image whose logical elements come from `owner_logical`
// and whose every non-logical padded element slot is forced to an all-ones
// code. Restricted to byte-aligned payloads, whose padding never shares a byte
// with a logical element.
[[nodiscard]] inline std::vector<std::byte> poisoned_standard_tiled_image(
        const iom::TensorSpec& owner_spec,
        std::span<const std::byte> owner_logical) {
    const std::size_t bits = bits_of(owner_spec.data_type);
    REQUIRE_EQ(bits % 8, std::size_t{0});
    const std::size_t carrier = bits / 8;
    REQUIRE_EQ(owner_logical.size(), owner_spec.logical_nbytes());
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t elements = owner_spec.shape.element_count();
    std::vector<std::byte> storage(
            canonical_padded_element_count(owner_spec) * carrier,
            std::byte{0xFF});
    std::vector<std::size_t> coordinates(dims.size());
    for (std::size_t linear = 0; linear < elements; ++linear) {
        std::size_t rest = linear;
        for (std::size_t axis = dims.size(); axis-- > 0;) {
            coordinates[axis] = rest % dims[axis];
            rest /= dims[axis];
        }
        const std::size_t slot = canonical_layout_slot(
                owner_spec, std::span<const std::size_t>{coordinates});
        std::memcpy(
                storage.data() + slot * carrier,
                owner_logical.data() + linear * carrier, carrier);
    }
    return storage;
}

// Expected raw output image of one applied request. For every logical view
// plane the reference decodes the fixture's own index bytes at the owner plane
// that view selects and copies that row's raw bits out of the separately
// encoded row-major table bytes.
[[nodiscard]] inline std::vector<std::byte> expected_embedding_output(
        const EmbeddingCase& item, const iom::TensorView& index_view,
        std::span<const std::byte> table_view_bytes,
        std::span<const std::byte> index_owner_bytes) {
    const std::size_t planes = embedding_view_planes(index_view);
    std::vector<std::uint64_t> codes(planes * item.run);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::size_t owner_plane =
                embedding_view_plane(index_view, plane);
        for (std::size_t row = 0; row < item.run; ++row) {
            codes[plane * item.run + row] = embedding_image_code(
                    index_owner_bytes, item.id_type,
                    owner_plane * item.run + row);
        }
    }
    return select_embedding_rows(
            table_view_bytes, item.payload, item.vocabulary, item.features,
            item.run, std::span<const std::uint64_t>{codes});
}

// ---------------------------------------------------------------------------
// Oracle self-check: explicit raw-code expectations and variant selectivity.
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::vector<std::byte> explicit_image(
        iom::DataType type, std::size_t elements,
        std::span<const std::uint64_t> codes) {
    const std::size_t bits = bits_of(type);
    REQUIRE_EQ(codes.size(), elements);
    std::vector<std::byte> image((elements * bits + 7) / 8, std::byte{0});
    for (std::size_t linear = 0; linear < elements; ++linear) {
        write_storage_bits(image.data(), linear * bits, bits, codes[linear]);
    }
    return image;
}

// Proves the reference against hand-authored raw-code rows, the deliberate
// negative variants, and their selectivity. Returns false when any expectation
// fails, so a driver can REQUIRE the self-check as one assertion.
[[nodiscard]] inline bool run_embedding_oracle_self_check() {
    bool ok = true;
    const auto select = [](std::span<const std::byte> table,
                           iom::DataType payload, std::size_t vocabulary,
                           std::size_t features, std::size_t run,
                           std::span<const std::uint64_t> codes,
                           EmbeddingOracleVariant variant) {
        return select_embedding_rows(
                table, payload, vocabulary, features, run, codes, variant);
    };
    const auto require_equal = [&ok](
                                       std::span<const std::byte> actual,
                                       std::span<const std::byte> expected,
                                       const char* what) {
        const bool same = actual.size() == expected.size()
                && std::equal(actual.begin(), actual.end(), expected.begin());
        CHECK_MESSAGE(same, what);
        ok = ok && same;
    };
    const auto require_different = [&ok](
                                           std::span<const std::byte> actual,
                                           std::span<const std::byte> expected,
                                           const char* what) {
        const bool differs = actual.size() != expected.size()
                || !std::equal(actual.begin(), actual.end(), expected.begin());
        CHECK_MESSAGE(differs, what);
        ok = ok && differs;
    };

    // Explicit BF16 fixture: V=4, F=3, explicit raw codes including signed
    // zero, both infinities, a NaN payload, a subnormal, and arbitrary
    // representable codes; the codes {3, 1, 1, 0} repeat a row and cover the
    // first and the last vocabulary row.
    const std::uint64_t table_codes[12] = {
            0x0000u, 0x8000u, 0x7F80u,
            0xFF80u, 0x7FC1u, 0x0001u,
            0x3F80u, 0xBF80u, 0x7BFFu,
            0x0001u, 0x80FFu, 0x1234u,
    };
    const std::uint64_t index_codes[4] = {3u, 1u, 1u, 0u};
    const std::uint64_t expected_codes[12] = {
            0x0001u, 0x80FFu, 0x1234u,
            0xFF80u, 0x7FC1u, 0x0001u,
            0xFF80u, 0x7FC1u, 0x0001u,
            0x0000u, 0x8000u, 0x7F80u,
    };
    {
        const std::vector<std::byte> table =
                explicit_image(iom::DataType::BF16, 12, table_codes);
        const std::vector<std::byte> expected =
                explicit_image(iom::DataType::BF16, 12, expected_codes);
        const std::vector<std::byte> honest = select(
                table, iom::DataType::BF16, 4, 3, 4, index_codes,
                EmbeddingOracleVariant::honest);
        require_equal(honest, expected, "BF16 raw-code row selection");
        require_different(
                select(table, iom::DataType::BF16, 4, 3, 4, index_codes,
                       EmbeddingOracleVariant::wrong_row),
                expected, "a wrong-row oracle is detected");
        require_different(
                select(table, iom::DataType::BF16, 4, 3, 4, index_codes,
                       EmbeddingOracleVariant::transposed),
                expected, "a transposed-axis oracle is detected");
        require_different(
                select(table, iom::DataType::BF16, 4, 3, 4, index_codes,
                       EmbeddingOracleVariant::padded_table),
                expected, "a padded-table oracle is detected");
        require_different(
                select(table, iom::DataType::BF16, 4, 3, 4, index_codes,
                       EmbeddingOracleVariant::numeric_reencode),
                expected, "a numeric re-encoding oracle is detected");
    }

    // Selectivity: with a single vocabulary row the wrong-row oracle coincides
    // with the reference, and with a tile-aligned feature extent the padded
    // table stride coincides as well, so the detections above are real
    // comparisons rather than an unconditional "differs".
    {
        const std::uint64_t single_codes[2] = {0x1234u, 0x5678u};
        const std::uint64_t single_index[2] = {0u, 0u};
        const std::vector<std::byte> single =
                explicit_image(iom::DataType::BF16, 2, single_codes);
        const std::vector<std::byte> honest = select(
                single, iom::DataType::BF16, 1, 2, 2, single_index,
                EmbeddingOracleVariant::honest);
        CHECK(honest == select(single, iom::DataType::BF16, 1, 2, 2,
                               single_index,
                               EmbeddingOracleVariant::wrong_row));
        CHECK(honest == select(single, iom::DataType::BF16, 1, 2, 2,
                               single_index,
                               EmbeddingOracleVariant::transposed));

        std::vector<std::uint64_t> aligned_codes(2 * 16);
        for (std::size_t linear = 0; linear < aligned_codes.size(); ++linear) {
            aligned_codes[linear] = splitmix64(linear + 7) & 0xFFFFu;
        }
        const std::uint64_t aligned_index[2] = {1u, 0u};
        const std::vector<std::byte> aligned = explicit_image(
                iom::DataType::BF16, aligned_codes.size(), aligned_codes);
        const std::vector<std::byte> aligned_honest = select(
                aligned, iom::DataType::BF16, 2, 16, 2, aligned_index,
                EmbeddingOracleVariant::honest);
        CHECK(aligned_honest
              == select(aligned, iom::DataType::BF16, 2, 16, 2, aligned_index,
                        EmbeddingOracleVariant::padded_table));

        const std::uint64_t plain_codes[4] = {0x3F80u, 0x4120u, 0xC120u,
                                              0x0000u};
        const std::uint64_t plain_index[2] = {1u, 0u};
        const std::vector<std::byte> plain =
                explicit_image(iom::DataType::BF16, 4, plain_codes);
        const std::vector<std::byte> plain_honest = select(
                plain, iom::DataType::BF16, 2, 2, 2, plain_index,
                EmbeddingOracleVariant::honest);
        CHECK(plain_honest
              == select(plain, iom::DataType::BF16, 2, 2, 2, plain_index,
                        EmbeddingOracleVariant::numeric_reencode));
    }

    // A leading-plane fixture: the plane-blind oracle must differ, and it must
    // coincide when every plane carries the same codes.
    {
        const std::uint64_t plane_table[8] = {
                0x0011u, 0x0022u, 0x0033u, 0x0044u,
                0x0055u, 0x0066u, 0x0077u, 0x0088u,
        };
        const std::uint64_t plane_codes[4] = {0u, 1u, 2u, 3u};
        const std::vector<std::byte> table =
                explicit_image(iom::DataType::BF16, 8, plane_table);
        const std::vector<std::byte> honest = select(
                table, iom::DataType::BF16, 4, 2, 2, plane_codes,
                EmbeddingOracleVariant::honest);
        require_different(
                select(table, iom::DataType::BF16, 4, 2, 2, plane_codes,
                       EmbeddingOracleVariant::plane_blind),
                honest, "a plane-blind oracle is detected");
        const std::uint64_t same_codes[4] = {2u, 3u, 2u, 3u};
        CHECK(select(table, iom::DataType::BF16, 4, 2, 2, same_codes,
                     EmbeddingOracleVariant::honest)
              == select(table, iom::DataType::BF16, 4, 2, 2, same_codes,
                        EmbeddingOracleVariant::plane_blind));
    }

    // Wide and sub-byte carriers with explicit raw codes, including integer
    // values above 2^53 and `U64_MAX`.
    {
        const std::uint64_t wide_table[6] = {
                0x0020000000000001ull, 0xFFFFFFFFFFFFFFFFull,
                0x0000000000000000ull, 0xFFFFFFFFFFFFF800ull,
                0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull,
        };
        const std::uint64_t wide_index[3] = {2u, 0u, 1u};
        const std::uint64_t wide_expected[6] = {
                0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull,
                0x0020000000000001ull, 0xFFFFFFFFFFFFFFFFull,
                0x0000000000000000ull, 0xFFFFFFFFFFFFF800ull,
        };
        const std::vector<std::byte> table =
                explicit_image(iom::DataType::U64, 6, wide_table);
        const std::vector<std::byte> expected =
                explicit_image(iom::DataType::U64, 6, wide_expected);
        require_equal(
                select(table, iom::DataType::U64, 3, 2, 3, wide_index,
                       EmbeddingOracleVariant::honest),
                expected, "U64 raw-code row selection above 2^53");
        require_equal(
                select(table, iom::DataType::F64, 3, 2, 3, wide_index,
                       EmbeddingOracleVariant::honest),
                expected, "F64 raw-code row selection of arbitrary codes");
    }
    {
        const std::uint64_t subbyte_table[8] = {
                0u, 1u, 2u, 3u,
                3u, 2u, 1u, 0u,
        };
        const std::uint64_t subbyte_index[2] = {1u, 0u};
        const std::uint64_t subbyte_expected[8] = {
                3u, 2u, 1u, 0u,
                0u, 1u, 2u, 3u,
        };
        const std::vector<std::byte> table =
                explicit_image(iom::DataType::I2, 8, subbyte_table);
        const std::vector<std::byte> expected =
                explicit_image(iom::DataType::I2, 8, subbyte_expected);
        require_equal(
                select(table, iom::DataType::I2, 2, 4, 2, subbyte_index,
                       EmbeddingOracleVariant::honest),
                expected, "I2 sub-byte raw-code row selection");

        const std::uint64_t bool_codes[4] = {0u, 1u, 1u, 0u};
        const std::uint64_t bool_index[2] = {1u, 0u};
        const std::uint64_t bool_expected[4] = {1u, 0u, 0u, 1u};
        const std::vector<std::byte> bool_table =
                explicit_image(iom::DataType::BOOL, 4, bool_codes);
        const std::vector<std::byte> bool_expected_image =
                explicit_image(iom::DataType::BOOL, 4, bool_expected);
        require_equal(
                select(bool_table, iom::DataType::BOOL, 2, 2, 2, bool_index,
                       EmbeddingOracleVariant::honest),
                bool_expected_image,
                "BOOL raw-code row selection of valid 0/1 codes");
    }

    return ok;
}

// ---------------------------------------------------------------------------
// Shared request cases.
// ---------------------------------------------------------------------------

struct EmbeddingLeadingCase {
    const char* label;
    std::vector<std::size_t> leading;
    std::size_t transform;
    std::size_t table_planes;
    bool poison_native_padding;
};

// The shared case matrix is the driver's own declared target matrix: a fixture
// may only instantiate leaves the port must reach (and, for a backend with a
// narrower storage table, exactly the leaves its device can store). The
// implemented flag decides for each generated case whether it is compared
// numerically or observed as a capability rejection.
[[nodiscard]] inline std::vector<EmbeddingCase> embedding_cases(
        std::span<const iom::DataType> payload_leaves,
        std::span<const iom::DataType> id_leaves) {
    REQUIRE_FALSE(payload_leaves.empty());
    REQUIRE_FALSE(id_leaves.empty());
    const iom::DataType default_payload =
            embedding_declares(payload_leaves, iom::DataType::BF16)
            ? iom::DataType::BF16
            : payload_leaves.front();
    const iom::DataType default_id =
            embedding_declares(id_leaves, iom::DataType::U32)
            ? iom::DataType::U32
            : id_leaves.back();
    std::vector<EmbeddingCase> cases;
    std::uint64_t salt = 0x1000;
    const auto add = [&](EmbeddingCase item) {
        item.salt = item.salt == 0 ? ++salt : item.salt;
        cases.push_back(std::move(item));
    };

    // Every declared payload leaf with U32 indices.
    for (const iom::DataType payload : payload_leaves) {
        EmbeddingCase item;
        item.label =
                "payload leaf " + std::to_string(static_cast<int>(payload));
        item.payload = payload;
        item.id_type = default_id;
        item.vocabulary = embedding_vocabulary_for(default_id, 17);
        add(std::move(item));
    }

    // Every declared integral index leaf with BF16 payload.
    for (const iom::DataType id_type : id_leaves) {
        EmbeddingCase item;
        item.label =
                "index leaf " + std::to_string(static_cast<int>(id_type));
        item.payload = default_payload;
        item.id_type = id_type;
        item.vocabulary = embedding_vocabulary_for(id_type, 17);
        add(std::move(item));
    }

    // Targeted mixed and wide combinations, without a full Cartesian matrix.
    const struct {
        const char* label;
        iom::DataType payload;
        iom::DataType id_type;
    } mixed[] = {
            {"mixed F64 payload with U64 indices", iom::DataType::F64,
             iom::DataType::U64},
            {"mixed I64 payload with I64 indices", iom::DataType::I64,
             iom::DataType::I64},
            {"mixed sub-byte U4 payload with I8 indices", iom::DataType::U4,
             iom::DataType::I8},
            {"mixed BOOL payload with U8 indices", iom::DataType::BOOL,
             iom::DataType::U8},
            {"mixed F6_E3M2 payload with U64 indices", iom::DataType::F6_E3M2,
             iom::DataType::U64},
    };
    for (const auto& entry : mixed) {
        if (!embedding_declares(payload_leaves, entry.payload)
                || !embedding_declares(id_leaves, entry.id_type)) {
            continue;
        }
        EmbeddingCase item;
        item.label = entry.label;
        item.payload = entry.payload;
        item.id_type = entry.id_type;
        item.vocabulary = embedding_vocabulary_for(entry.id_type, 17);
        add(std::move(item));
    }

    // Run boundaries.
    for (const std::size_t run : {1u, 15u, 16u, 17u}) {
        EmbeddingCase item;
        item.label = "run " + std::to_string(run);
        item.run = run;
        item.payload = default_payload;
        item.id_type = default_id;
        item.vocabulary = embedding_vocabulary_for(default_id, 17);
        add(std::move(item));
    }

    // Non-tile feature sizes.
    for (const std::size_t features : {1u, 15u, 17u, 31u, 33u}) {
        EmbeddingCase item;
        item.label = "features " + std::to_string(features);
        item.features = features;
        item.payload = default_payload;
        item.id_type = default_id;
        item.vocabulary = embedding_vocabulary_for(default_id, 17);
        add(std::move(item));
    }

    // Vocabulary boundaries, clamped into the declared index leaf's own
    // nonnegative domain.
    for (const std::size_t vocabulary : {1u, 16u, 17u, 33u}) {
        EmbeddingCase item;
        item.label = "vocabulary " + std::to_string(vocabulary);
        item.payload = default_payload;
        item.id_type = default_id;
        item.vocabulary = embedding_vocabulary_for(default_id, vocabulary);
        add(std::move(item));
    }

    // Independent leading planes through rank eight, including rank-two
    // selected table views over a larger owner and poisoned native padding.
    const std::vector<EmbeddingTransform> transforms = embedding_transforms();
    const EmbeddingLeadingCase leading_cases[] = {
            {"rank two full", {}, 0, 1, false},
            {"rank three full", {2}, 0, 1, false},
            {"rank three select", {2}, 1, 1, false},
            {"rank three stepped slice with a selected table plane", {2}, 2, 2,
             true},
            {"rank four reversed permutation", {2, 2}, 3, 1, false},
            {"rank four single leading axis reshape", {2, 2}, 4, 1, false},
            {"rank four nested transform", {2, 2}, 5, 1, false},
            {"rank five full", {2, 2, 2}, 0, 1, false},
            {"rank five nested transform", {2, 2, 2}, 5, 1, false},
            {"rank six full with a selected table plane", {2, 3, 2, 2}, 0, 2,
             true},
            {"rank six reversed permutation", {2, 3, 2, 2}, 3, 1, false},
            {"rank seven stepped slice", {2, 2, 2, 2, 2}, 2, 1, false},
            {"rank seven single leading axis reshape", {2, 2, 2, 2, 2}, 4, 2,
             true},
            {"rank eight full with a selected table plane", {2, 2, 2, 2, 2, 2},
             0, 2, true},
            {"rank eight nested transform", {2, 2, 2, 2, 2, 2}, 5, 1, false},
    };
    for (const EmbeddingLeadingCase& entry : leading_cases) {
        EmbeddingCase item;
        item.label = std::string(entry.label) + " [rank " +
                std::to_string(entry.leading.size() + 2) + ", " +
                transforms[entry.transform].label + "]";
        item.leading_dims = entry.leading;
        item.transform = transforms[entry.transform].build;
        item.table_planes = entry.table_planes;
        item.poison_native_padding = entry.poison_native_padding;
        item.payload = default_payload;
        item.id_type = default_id;
        item.vocabulary = embedding_vocabulary_for(default_id, 17);
        add(std::move(item));
    }

    // Accepted out-of-vocabulary and negative-ID fixtures: one per index leaf,
    // with a vocabulary small enough that the positive code stays representable
    // without narrowing. The output is unspecified and is never compared.
    for (const iom::DataType id_type : id_leaves) {
        EmbeddingCase item;
        item.label = "out-of-vocabulary index leaf " +
                std::to_string(static_cast<int>(id_type));
        item.payload = default_payload;
        item.id_type = id_type;
        item.vocabulary = embedding_vocabulary_for(id_type, 5);
        item.run = 4;
        const EmbeddingOovCodes codes =
                embedding_oov_codes(id_type, item.vocabulary);
        item.out_of_vocabulary = true;
        item.code = [codes](std::size_t, std::size_t row) {
            switch (row) {
                case 1:
                    return codes.positive;
                case 2:
                    return codes.maximum;
                case 3:
                    return codes.has_negative ? codes.negative
                                              : codes.maximum;
                default:
                    return std::uint64_t{0};
            }
        };
        add(std::move(item));
    }

    return cases;
}

// ---------------------------------------------------------------------------
// Common embedding queue: the shared admission, ownership, queue-order, and
// failure double. It exercises the common `DeviceOps` embedding path exactly as
// the existing common binary queue exercises the binary one, and it never
// touches native compute or native storage.
// ---------------------------------------------------------------------------

// One live raw-workspace owner over a fabricated, caller-selected address.
// Workspace liveness, alignment, size, device, and range rules are all
// observable through it without allocating native scratch.
class ConformanceWorkspace final : public iom::RawWorkspace {
public:
    ConformanceWorkspace(iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

// A view whose owning workspace is already destroyed: only the owner identity
// survives, which is exactly what the liveness rule observes.
[[nodiscard]] inline iom::RawWorkspaceView dead_workspace_view(
        iom::Device& device, void* address, std::size_t bytes) {
    ConformanceWorkspace dead(device, address, bytes);
    return dead.view();
}

class CommonEmbeddingQueue final : public iom::DeviceOps {
public:
    enum class Failure { none, post_acceptance };

    struct EmbeddingRecord {
        std::uint64_t sequence = 0;
        iom::detail::BinaryEntryRegistration entries;
        iom::detail::WorkspaceLease workspace_lease;
        bool retained_failure = false;
    };

    CommonEmbeddingQueue(
            const iom::Device& device,
            iom::WorkspaceRequirements requirements)
            : iom::DeviceOps(device), requirements_(requirements) {}

    void inject_failure(Failure failure) noexcept {
        next_failure_ = failure;
    }

    [[nodiscard]] const std::vector<EmbeddingRecord>& embedding_records()
            const noexcept {
        return records_;
    }

    [[nodiscard]] const std::vector<std::string_view>& submissions()
            const noexcept {
        return submissions_;
    }

    [[nodiscard]] std::size_t registered_at(const void* address) const {
        return registry_.registry
                .snapshot_for(const_cast<void*>(address))
                .size();
    }

    // Plays the in-order completion of one accepted submission: an embedding
    // record releases or invalidates its owners and completes its workspace
    // lease exactly as a proven-completion worker would, and a view-less copy
    // or probe submission only completes its sequence. An unknown sequence is
    // still a contract violation.
    void finish(std::uint64_t sequence) {
        for (const EmbeddingRecord& record : records_) {
            if (record.sequence != sequence) {
                continue;
            }
            (void)iom::detail::release_or_invalidate_binary_entries(
                    registry_.registry, record.entries,
                    record.retained_failure, !record.retained_failure);
            iom::detail::complete_workspace_lease(
                    registry_, record.workspace_lease, true);
            complete(sequence);
            return;
        }
        if (std::find(plain_.begin(), plain_.end(), sequence)
                != plain_.end()) {
            complete(sequence);
            return;
        }
        throw std::invalid_argument("unknown common embedding sequence");
    }

    // A view-less submission that observes queue identity and sequence
    // allocation without reusing an accepted operation.
    iom::oid probe() {
        const iom::oid token = submit([this](std::uint64_t) {
            submissions_.push_back("probe");
        });
        plain_.push_back(token_sequence(token));
        return token;
    }

protected:
    iom::oid copy_impl(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument(
                    "common embedding queue copy needs identical specs");
        }
        const iom::oid token = submit([this](std::uint64_t) {
            submissions_.push_back("copy");
        });
        plain_.push_back(token_sequence(token));
        return token;
    }

    // The pure requirement hook reports this declaration's exact policy and
    // touches no view metadata, queue state, or allocation.
    iom::WorkspaceRequirements embedding_workspace_requirements_impl(
            const iom::TensorView&, const iom::TensorView&,
            const iom::TensorView&) override {
        return requirements_;
    }

    iom::oid embedding_impl(const EmbeddingRequest& request) override {
        const Failure failure = std::exchange(next_failure_, Failure::none);
        iom::detail::Fence fence;
        fence.invoke = [](const iom::detail::Fence&) noexcept {
            return iom::detail::FenceResult::pending();
        };
        return submit_embedding(
                request, registry_, queue_id_, fence,
                [this, failure](
                        std::uint64_t sequence,
                        const EmbeddingRequest& snapshot,
                        iom::detail::BinaryEntryRegistration entries) {
                    records_.push_back(EmbeddingRecord{
                            sequence, entries, snapshot.workspace_lease,
                            failure == Failure::post_acceptance});
                    submissions_.push_back("embedding");
                    if (failure == Failure::post_acceptance) {
                        commit_failure(
                                sequence,
                                std::make_exception_ptr(
                                        std::invalid_argument(
                                                "common embedding invalid "
                                                "index")));
                    }
                });
    }

private:
    iom::detail::RegistryState registry_;
    iom::detail::QueueId queue_id_ =
            iom::detail::allocate_queue_id(registry_);
    iom::WorkspaceRequirements requirements_{};
    Failure next_failure_ = Failure::none;
    std::vector<EmbeddingRecord> records_;
    std::vector<std::string_view> submissions_;
    // Sequences of the view-less copy and probe submissions, which own no
    // operand, lease, or registry entry to release.
    std::vector<std::uint64_t> plain_;
};

// Requires a deferred embedding data failure to rethrow `std::invalid_argument`
// on every repeat wait, without pinning the message wording.
inline void expect_repeated_invalid_argument(
        iom::DeviceOps& queue, iom::oid token) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        CHECK_THROWS_AS(queue.wait(token), std::invalid_argument);
    }
}

// Arms one case's observer window and disarms it on every exit path, including
// a failing assertion and a thrown exception. Each case declares its window
// after its tensors and before its queue, so teardown always runs in this
// order: queue drain, window close, then tensor destruction. A failing
// assertion therefore never leaves the traffic gate armed across tensor
// destruction, and no teardown diagnostic can mask the real failure.
class EmbeddingCaseWindow final {
public:
    explicit EmbeddingCaseWindow(ConformanceObserver* observer)
            : observer_(observer) {
        if (observer_ != nullptr) {
            observer_->setup_complete();
        }
    }
    ~EmbeddingCaseWindow() {
        if (observer_ != nullptr) {
            observer_->case_complete();
        }
    }
    EmbeddingCaseWindow(const EmbeddingCaseWindow&) = delete;
    EmbeddingCaseWindow& operator=(const EmbeddingCaseWindow&) = delete;
    EmbeddingCaseWindow(EmbeddingCaseWindow&&) = delete;
    EmbeddingCaseWindow& operator=(EmbeddingCaseWindow&&) = delete;

private:
    ConformanceObserver* observer_;
};

// Qualifies the live owner view specifications of one request for the
// recognized non-NONE quantization probe and restores them to `NONE` when its
// scope ends, on every exit path — including a failing assertion and a thrown
// exception — so no operand is ever destroyed while its specification carries
// a recognized grouped quantization format.
//
// Two facts force this shape. A view copy qualified on its own is
// admission-invalid, because a view/owner specification mismatch is rejected
// before the capability decision, so the owner view itself must carry the
// format. And a qualified specification cannot survive teardown on every
// backend: `TensorSpec::tiled_storage_nbytes()` rejects grouped formats, and
// the CUDA, ROCm, and SYCL tensor destructors derive their storage extent from
// the owner specification inside a `noexcept` destructor, so an unrestored
// qualification aborts the process instead of reporting the assertion. The
// restore is therefore scoped, backend-independent, and complete before any
// operand is destroyed.
class EmbeddingQuantizationQualification final {
public:
    EmbeddingQuantizationQualification(
            iom::Tensor& table, iom::Tensor& indices, iom::Tensor& out)
            : specs_{qualify(table), qualify(indices), qualify(out)} {}

    ~EmbeddingQuantizationQualification() {
        for (iom::TensorSpec* spec : specs_) {
            spec->quantization = iom::QuantizationFormat::NONE;
        }
    }

    EmbeddingQuantizationQualification(
            const EmbeddingQuantizationQualification&) = delete;
    EmbeddingQuantizationQualification& operator=(
            const EmbeddingQuantizationQualification&) = delete;
    EmbeddingQuantizationQualification(
            EmbeddingQuantizationQualification&&) = delete;
    EmbeddingQuantizationQualification& operator=(
            EmbeddingQuantizationQualification&&) = delete;

private:
    // Qualifies one live owner view specification and returns it.
    [[nodiscard]] static iom::TensorSpec* qualify(iom::Tensor& owner) {
        iom::TensorSpec& spec =
                const_cast<iom::TensorSpec&>(owner.view().spec());
        spec.quantization = iom::QuantizationFormat::OCP_MXFP4;
        return &spec;
    }

    std::array<iom::TensorSpec*, 3> specs_;
};

// ---------------------------------------------------------------------------
// Shared case runners.
// ---------------------------------------------------------------------------

// One submission on the real queue with the declared workspace policy: exactly
// the queried requirement, provisioned from the candidate device. A
// requirement the declaration did not predict fails here instead of silently
// changing the submission.
struct EmbeddingRealSubmission {
    iom::oid result = 0;
    std::unique_ptr<iom::RawWorkspace> workspace;
};

[[nodiscard]] inline EmbeddingRealSubmission submit_real_embedding(
        iom::Device& device, iom::DeviceOps& queue,
        const EmbeddingDeclaration& declaration, const iom::TensorView& table,
        const iom::TensorView& indices, iom::TensorView& out) {
    const iom::WorkspaceRequirements requirements =
            queue.embedding_workspace_requirements(table, indices, out);
    CHECK_EQ(requirements.bytes, declaration.workspace.bytes);
    CHECK_EQ(requirements.alignment, declaration.workspace.alignment);
    EmbeddingRealSubmission submission;
    if (requirements.bytes == 0) {
        submission.result = queue.embedding(table, indices, out);
        return submission;
    }
    submission.workspace = device.create_workspace(requirements.bytes);
    const iom::RawWorkspaceView workspace = submission.workspace->view();
    submission.result = queue.embedding(table, indices, out, workspace);
    return submission;
}

// The shared request cases through the candidate's real queue. Every case
// builds its exact fixture, seeds it, submits one request with the declared
// workspace policy, and either compares the output bit-for-bit against the
// independent reference or asserts capability rejection with an unchanged
// output. A declared leaf that reports `Unsupported`, or an undeclared leaf
// that is accepted, fails here: neither is gather conformance.
inline void run_embedding_reference_conformance(
        const ConformanceDevices& devices,
        const EmbeddingDeclaration& declaration,
        ConformanceObserver* observer = nullptr,
        AcceleratorStorageOracle* oracle = nullptr) {
    iom::Device& candidate = devices.candidate;
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);

    // The declaration must stay inside the matrix it declares as its target,
    // every implemented index leaf must be one of the twelve contract leaves,
    // and the declared target matrix must be storable by the candidate device:
    // the suite's fixtures are exactly that matrix. An unported port declares
    // the empty implemented span.
    for (const iom::DataType leaf : declaration.implemented_payloads) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(embedding_declares(declaration.target_payloads, leaf));
    }
    for (const iom::DataType leaf : declaration.implemented_ids) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(embedding_declares(declaration.target_ids, leaf));
        CHECK(embedding_declares(kEmbeddingIdSpan, leaf));
    }
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    for (const iom::DataType leaf : declaration.target_payloads) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(embedding_declares(storable, leaf));
    }
    for (const iom::DataType leaf : declaration.target_ids) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(embedding_declares(storable, leaf));
    }

    const std::vector<EmbeddingCase> cases = embedding_cases(
            declaration.target_payloads, declaration.target_ids);
    for (const EmbeddingCase& item : cases) {
        CAPTURE(item.label);
        const EmbeddingCaseSpecs specs = embedding_case_specs(item);
        auto table_owner = candidate.create_tensor(specs.table_owner);
        auto index_owner = candidate.create_tensor(specs.index_owner);
        auto out_owner = candidate.create_tensor(specs.out_owner);
        const iom::TensorView table_view = item.table_planes > 1
                ? table_owner->view().select(0, item.table_planes - 1)
                : iom::TensorView{table_owner->view()};
        const iom::TensorView index_view = item.transform
                ? item.transform(index_owner->view())
                : iom::TensorView{index_owner->view()};
        iom::TensorView out_view = item.transform
                ? item.transform(out_owner->view())
                : iom::TensorView{out_owner->view()};

        const std::vector<std::byte> table_view_bytes =
                encode_logical(table_view.spec(), item.salt);
        const std::vector<std::byte> table_poison = encode_logical(
                specs.table_owner, item.salt ^ 0xAE5Eull);
        const std::vector<std::byte> out_poison = encode_logical(
                specs.out_owner, item.salt ^ 0x1D0Full);
        const std::vector<std::byte> out_view_poison =
                embedding_view_image(out_view, out_poison);
        std::function<std::uint64_t(std::size_t, std::size_t)> codes;
        if (item.code) {
            codes = item.code;
        } else {
            codes = [&item](std::size_t plane, std::size_t row) {
                return embedding_valid_code(
                        item.vocabulary, item.run, plane, row, item.salt);
            };
        }
        const std::vector<std::byte> index_owner_bytes =
                encode_index_owner(specs.index_owner, item.run, codes);
        const std::vector<std::byte> table_owner_bytes =
                table_owner_image(
                        specs.table_owner, table_view, table_view_bytes,
                        table_poison);

        const EmbeddingCaseWindow window(observer);
        const bool poison_native_padding = item.poison_native_padding
                && oracle != nullptr && bits_of(item.payload) % 8 == 0;
        if (poison_native_padding) {
            oracle->set_owner_spec(specs.table_owner);
            oracle->seed(
                    table_owner->view(),
                    poisoned_standard_tiled_image(
                            specs.table_owner, table_owner_bytes));
        } else {
            copy_from_host(table_owner->view(), table_owner_bytes);
        }
        copy_from_host(index_owner->view(), index_owner_bytes);
        copy_from_host(out_owner->view(), out_poison);
        auto queue = candidate.create_ops();

        const bool implemented =
                embedding_implements(declaration, item.payload, item.id_type);
        if (!implemented) {
            // An undeclared leaf pair is a capability rejection: a negative
            // synchronous result with no output effect that never inspects
            // unusable scratch.
            CHECK_THROWS_AS(
                    (void)queue->embedding_workspace_requirements(
                            table_view, index_view, out_view),
                    std::runtime_error);
            CHECK_EQ(
                    queue->embedding(table_view, index_view, out_view),
                    unsupported);
            require_logical_bytes(
                    out_view, out_view_poison,
                    item.label + ": rejection changed the output");
            require_logical_bytes(
                    table_view, table_view_bytes,
                    item.label + ": rejection changed the table");
            continue;
        }

        const EmbeddingRealSubmission submission = submit_real_embedding(
                candidate, *queue, declaration, table_view, index_view,
                out_view);
        REQUIRE(iom::oid_is_token(submission.result));
        if (item.out_of_vocabulary) {
            // Queued index data: an out-of-vocabulary or negative ID is a
            // positive accepted token and a data failure, never a host
            // admission scan. The output is unspecified and is not compared.
            expect_repeated_invalid_argument(*queue, submission.result);
        } else {
            CHECK_NOTHROW(queue->wait(submission.result));
            const std::vector<std::byte> expected = expected_embedding_output(
                    item, index_view, table_view_bytes, index_owner_bytes);
            require_logical_bytes(
                    out_view, expected, item.label + ": gather output");
            // Every unselected output owner plane still holds its seeded
            // poison, so a gather that writes outside its own logical window or
            // consumes uninitialized bytes fails here.
            const std::vector<std::byte> observed_owner =
                    read_logical(out_owner->view());
            const std::size_t view_planes = embedding_view_planes(out_view);
            const std::size_t owner_planes =
                    embedding_view_planes(out_owner->view());
            const std::size_t plane_elements =
                    embedding_plane_elements(out_view);
            for (std::size_t owner_plane = 0; owner_plane < owner_planes;
                 ++owner_plane) {
                std::optional<std::size_t> view_plane;
                for (std::size_t plane = 0; plane < view_planes; ++plane) {
                    if (embedding_view_plane(out_view, plane) == owner_plane) {
                        view_plane = plane;
                    }
                }
                for (std::size_t element = 0; element < plane_elements;
                     ++element) {
                    const std::size_t linear =
                            owner_plane * plane_elements + element;
                    const std::uint64_t observed = embedding_image_code(
                            observed_owner, item.payload, linear);
                    const std::uint64_t reference = view_plane.has_value()
                            ? embedding_image_code(
                                      expected, item.payload,
                                      view_plane.value() * plane_elements
                                              + element)
                            : embedding_image_code(
                                      out_poison, item.payload, linear);
                    CHECK_MESSAGE(
                            observed == reference,
                            item.label << ": output owner plane "
                                       << owner_plane << " element "
                                       << element
                                       << " differs from the independent "
                                          "reference");
                }
            }
            require_logical_bytes(
                    table_view, table_view_bytes,
                    item.label + ": gather changed the table");
        }
    }

    // A structurally invalid request keeps its own category before the
    // capability decision, and input/input read aliasing stays allowed: it is
    // never an admission rejection.
    {
        auto table = candidate.create_tensor(
                embedding_spec({1, 8}, iom::DataType::U32));
        auto out = candidate.create_tensor(
                embedding_spec({8, 8}, iom::DataType::U32));
        auto row_index = candidate.create_tensor(
                embedding_spec({2, 8}, iom::DataType::U32));
        auto wide_table = candidate.create_tensor(
                embedding_spec({2, 8, 8}, iom::DataType::U32));
        auto foreign = devices.foreign.create_tensor(
                embedding_spec({1, 8}, iom::DataType::U32));
        const EmbeddingCaseWindow window(observer);
        auto queue = candidate.create_ops();
        // The index view must keep its single dummy row axis and the table
        // view must be rank two.
        CHECK_EQ(
                queue->embedding(
                        table->view(), row_index->view(), out->view()),
                invalid);
        CHECK_EQ(
                queue->embedding(
                        wide_table->view(), table->view(), out->view()),
                invalid);
        // A view from another device instance is never an operand.
        CHECK_EQ(
                queue->embedding(
                        foreign->view(), table->view(), out->view()),
                invalid);
        // Input/input read aliasing is allowed. Only the capability decision
        // may follow for an undeclared leaf pair.
        const iom::oid aliased =
                queue->embedding(table->view(), table->view(), out->view());
        CHECK_NE(aliased, invalid);
        if (!embedding_implements(
                    declaration, iom::DataType::U32, iom::DataType::U32)) {
            CHECK_EQ(aliased, unsupported);
        }
    }
}

// The common admission, ownership, workspace, queue-order, and failure cases
// through the shared embedding double, on the candidate's own tensors. These
// observe the common `DeviceOps` contract every port inherits; they never claim
// native numerical conformance.
inline void run_embedding_common_conformance(
        const ConformanceDevices& devices,
        const EmbeddingDeclaration& declaration,
        ConformanceObserver* observer = nullptr) {
    iom::Device& candidate = devices.candidate;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid exhausted = iom::to_oid(iom::OidError::ResourceExhausted);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);

    // The pure requirement query: the declared policy verbatim, validated with
    // the submission's own categories, and with no record, submission,
    // registration, lease, or sequence consumption.
    {
        auto table = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto wrong_rows = candidate.create_tensor(
                embedding_spec({2, 17}, iom::DataType::U32));
        auto floating_ids = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::F32));
        auto foreign = devices.foreign.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        const EmbeddingCaseWindow window(observer);
        CommonEmbeddingQueue queue(candidate, declaration.workspace);
        const iom::WorkspaceRequirements requirements =
                queue.embedding_workspace_requirements(
                        table->view(), indices->view(), out->view());
        CHECK_EQ(requirements.bytes, declaration.workspace.bytes);
        CHECK_EQ(requirements.alignment, declaration.workspace.alignment);
        CHECK(queue.embedding_workspace_requirements(
                      table->view(), indices->view(), out->view())
              == declaration.workspace);
        CHECK_THROWS_AS(
                (void)queue.embedding_workspace_requirements(
                        foreign->view(), indices->view(), out->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.embedding_workspace_requirements(
                        table->view(), wrong_rows->view(), out->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.embedding_workspace_requirements(
                        table->view(), floating_ids->view(), out->view()),
                std::runtime_error);
        CHECK(queue.embedding_records().empty());
        CHECK(queue.submissions().empty());
        CHECK_EQ(
                queue.registered_at(table->view().native_handle()),
                std::size_t{0});
        CHECK_EQ(
                queue.registered_at(indices->view().native_handle()),
                std::size_t{0});
        CHECK_EQ(
                queue.registered_at(out->view().native_handle()),
                std::size_t{0});
        // The consumed token sequence proves that no query submitted anything.
        CHECK_EQ(token_sequence(queue.probe()), 1);
        CHECK_EQ(queue.submissions().size(), std::size_t{1});
    }

    // Common admission accepts every declared payload leaf and every integral
    // index leaf, and rejects the recognized but inapplicable index leaves as
    // `Unsupported`. A common-layer rejection of a valid leaf is observable
    // here for every backend, and never claims numerical conformance.
    {
        ConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5100), 64);
        // The canonical pairing of the shared matrix: U32 indices with every
        // payload leaf and the BF16 payload with every index leaf, falling
        // back to a declared leaf when a driver's matrix omits it.
        const iom::DataType declared_id =
                embedding_declares(declaration.target_ids, iom::DataType::U32)
                ? iom::DataType::U32
                : declaration.target_ids.back();
        const iom::DataType declared_payload =
                embedding_declares(
                        declaration.target_payloads, iom::DataType::BF16)
                ? iom::DataType::BF16
                : declaration.target_payloads.front();
        for (const iom::DataType payload : declaration.target_payloads) {
            CAPTURE(static_cast<int>(payload));
            auto table = candidate.create_tensor(
                    embedding_spec({17, 17}, payload));
            auto indices = candidate.create_tensor(
                    embedding_spec({1, 17}, declared_id));
            auto out = candidate.create_tensor(
                    embedding_spec({17, 17}, payload));
            const EmbeddingCaseWindow window(observer);
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            const iom::oid token = declaration.workspace.bytes == 0
                    ? queue.embedding(
                              table->view(), indices->view(), out->view())
                    : queue.embedding(
                              table->view(), indices->view(), out->view(),
                              scratch.view().subrange(
                                      0, declaration.workspace.bytes));
            REQUIRE(iom::oid_is_token(token));
            REQUIRE_EQ(queue.embedding_records().size(), std::size_t{1});
            CHECK_EQ(
                    queue.embedding_records().back().entries.count,
                    std::size_t{3});
            // The three live operand owners are retained until proven
            // completion and released exactly then.
            CHECK_EQ(
                    queue.registered_at(table->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(indices->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(out->view().native_handle()),
                    std::size_t{1});
            queue.finish(token_sequence(token));
            CHECK_NOTHROW(queue.wait(token));
            CHECK_EQ(
                    queue.registered_at(table->view().native_handle()),
                    std::size_t{0});
        }
        for (const iom::DataType id_type : declaration.target_ids) {
            CAPTURE(static_cast<int>(id_type));
            auto table = candidate.create_tensor(
                    embedding_spec({17, 17}, declared_payload));
            auto indices = candidate.create_tensor(
                    embedding_spec({1, 17}, id_type));
            auto out = candidate.create_tensor(
                    embedding_spec({17, 17}, declared_payload));
            const EmbeddingCaseWindow window(observer);
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            const iom::oid token = declaration.workspace.bytes == 0
                    ? queue.embedding(
                              table->view(), indices->view(), out->view())
                    : queue.embedding(
                              table->view(), indices->view(), out->view(),
                              scratch.view().subrange(
                                      0, declaration.workspace.bytes));
            REQUIRE(iom::oid_is_token(token));
            queue.finish(token_sequence(token));
            CHECK_NOTHROW(queue.wait(token));
        }
        // Recognized but inapplicable index leaves: the operation capability
        // stage reports them as `Unsupported` with no owner registration, no
        // record, and no sequence consumption. Only leaves the device can
        // store are instantiated.
        for (const iom::DataType id_type : embedding_storable(
                     kEmbeddingNonIdSpan, candidate.supported_data_types())) {
            CAPTURE(static_cast<int>(id_type));
            auto table = candidate.create_tensor(
                    embedding_spec({17, 17}, declared_payload));
            auto indices = candidate.create_tensor(
                    embedding_spec({1, 17}, id_type));
            auto out = candidate.create_tensor(
                    embedding_spec({17, 17}, declared_payload));
            const EmbeddingCaseWindow window(observer);
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view()),
                    unsupported);
            CHECK(queue.embedding_records().empty());
        }
    }

    // Structural admission, checked arithmetic, device identity, and the
    // conservative output/input overlap rule, all with their own categories and
    // none of them consuming a sequence or registering an owner.
    {
        auto table = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto wrong_run = candidate.create_tensor(
                embedding_spec({16, 17}, iom::DataType::BF16));
        auto wrong_leading = candidate.create_tensor(
                embedding_spec({2, 1, 17}, iom::DataType::U32));
        auto wrong_leading_out = candidate.create_tensor(
                embedding_spec({3, 17, 17}, iom::DataType::BF16));
        auto other_payload = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::F16));
        auto rank_three_table = candidate.create_tensor(
                embedding_spec({2, 17, 17}, iom::DataType::BF16));
        auto foreign_table = devices.foreign.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto shared = candidate.create_tensor(
                embedding_spec({2, 17, 17}, iom::DataType::BF16));
        auto wide_index_owner = candidate.create_tensor(
                embedding_spec({3, 1, 17}, iom::DataType::U32));
        auto wide_out = candidate.create_tensor(
                embedding_spec({3, 17, 17}, iom::DataType::BF16));
        // Operands of the recognized non-NONE quantization probe: they are
        // created before this case's gated window, because tensor storage is
        // never allocated inside a transfer or operation.
        auto grouped_table = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto grouped_indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto grouped_out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        const EmbeddingCaseWindow window(observer);
        CommonEmbeddingQueue queue(candidate, declaration.workspace);
        const auto reject = [&queue, invalid](
                                    const iom::TensorView& table_view,
                                    const iom::TensorView& index_view,
                                    iom::TensorView& out_view,
                                    iom::oid expected, const char* what) {
            const iom::oid result =
                    queue.embedding(table_view, index_view, out_view);
            CHECK_MESSAGE(result == expected, what);
            CHECK(queue.embedding_records().empty());
        };
        reject(rank_three_table->view(), indices->view(), out->view(), invalid,
               "a rank-three table view is rejected");
        reject(table->view(), out->view(), out->view(), invalid,
               "an index view without its single dummy row axis is rejected");
        reject(table->view(), indices->view(), wrong_run->view(), invalid,
               "an output run axis that differs from the index run axis is "
               "rejected");
        reject(table->view(), wrong_leading->view(),
               wrong_leading_out->view(), invalid,
               "index and output leading tuples must match exactly");
        reject(table->view(), indices->view(), other_payload->view(), invalid,
               "table and output must share one leaf encoding");
        reject(foreign_table->view(), indices->view(), out->view(), invalid,
               "a view from another device instance is rejected");
        // Output/input aliasing is rejected conservatively on owner identity
        // even where the selected windows appear disjoint.
        iom::TensorView aliased_out = shared->view().select(0, 0);
        reject(shared->view().select(0, 1), indices->view(), aliased_out,
               invalid,
               "an output view of the table's own owner is rejected");
        {
            iom::TensorView unknown_type = table->view();
            const_cast<iom::TensorSpec&>(unknown_type.spec()).data_type =
                    static_cast<iom::DataType>(127);
            reject(unknown_type, indices->view(), out->view(), invalid,
                   "an unknown payload encoding is rejected");
            iom::TensorView zero_extent = table->view();
            const_cast<std::size_t*>(
                    zero_extent.spec().shape.dimensions().data())[0] = 0;
            reject(zero_extent, indices->view(), out->view(), invalid,
                   "a zero runtime extent is rejected");
        }
        {
            // Recognized non-NONE quantization is a capability rejection, so
            // this fixture qualifies all three operands with the same
            // recognized format. Qualifying fewer operands would be a
            // table/output specification mismatch — an admission error that
            // precedes the capability decision — and would therefore observe
            // the wrong contract rule, and a qualification on a bare view copy
            // is itself admission-invalid. The qualification is scoped to this
            // block and removed again on every exit path, before any operand is
            // destroyed. These operands were created before this case's gated
            // window, so qualifying them allocates nothing.
            const EmbeddingQuantizationQualification qualified(
                    *grouped_table, *grouped_indices, *grouped_out);
            iom::TensorView grouped_out_view = grouped_out->view();
            reject(grouped_table->view(), grouped_indices->view(),
                   grouped_out_view, unsupported,
                   "recognized non-NONE quantization is unsupported");
        }
        {
            // Checked plane-addressing arithmetic overflows instead of
            // wrapping, and the report stays the established Overflow category.
            iom::TensorView wide_index = wide_index_owner->view();
            const_cast<std::size_t*>(wide_index.plane_strides().data())[0] =
                    std::numeric_limits<std::size_t>::max();
            CHECK_EQ(
                    queue.embedding(
                            table->view(), wide_index, wide_out->view()),
                    iom::to_oid(iom::OidError::Overflow));
        }
        CHECK_EQ(token_sequence(queue.probe()), 1);
    }

    // The declared workspace policy: a zero requirement accepts and ignores any
    // supplied range, while a positive requirement rejects every unusable range
    // before dispatch, leases exactly the live disjoint range, and never reuses
    // a range whose completion is unproven.
    {
        auto table = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        ConformanceWorkspace aligned(
                candidate, reinterpret_cast<void*>(0x5200), 64);
        ConformanceWorkspace small(
                candidate, reinterpret_cast<void*>(0x5300), 16);
        ConformanceWorkspace misaligned(
                candidate, reinterpret_cast<void*>(0x5401), 64);
        ConformanceWorkspace foreign(
                devices.foreign, reinterpret_cast<void*>(0x5500), 64);
        ConformanceWorkspace overlapping(
                candidate, table->view().native_handle(), 64);
        const iom::RawWorkspaceView dead = dead_workspace_view(
                candidate, reinterpret_cast<void*>(0x5600), 64);
        const EmbeddingCaseWindow window(observer);
        CommonEmbeddingQueue queue(candidate, declaration.workspace);
        if (declaration.workspace.bytes == 0) {
            // A zero requirement neither validates nor leases an unused range:
            // every supplied view is accepted and nothing is registered.
            const iom::oid tokens[6] = {
                    queue.embedding(
                            table->view(), indices->view(), out->view()),
                    queue.embedding(
                            table->view(), indices->view(), out->view(), dead),
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            foreign.view()),
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            small.view()),
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            misaligned.view()),
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            overlapping.view()),
            };
            for (const iom::oid token : tokens) {
                REQUIRE(iom::oid_is_token(token));
            }
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5200)),
                    std::size_t{0});
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5300)),
                    std::size_t{0});
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5500)),
                    std::size_t{0});
            for (const iom::oid token : tokens) {
                queue.finish(token_sequence(token));
                CHECK_NOTHROW(queue.wait(token));
            }
        } else {
            const std::size_t bytes = declaration.workspace.bytes;
            const iom::RawWorkspaceView usable =
                    aligned.view().subrange(0, bytes);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view()),
                    invalid);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(), dead),
                    invalid);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            foreign.view()),
                    invalid);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            small.view()),
                    invalid);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            misaligned.view()),
                    invalid);
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            overlapping.view()),
                    invalid);
            CHECK(queue.embedding_records().empty());

            const iom::oid leased = queue.embedding(
                    table->view(), indices->view(), out->view(), usable);
            REQUIRE(iom::oid_is_token(leased));
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5200)),
                    std::size_t{1});
            // A live overlapping range is bounded-resource exhaustion and
            // leaves neither record nor lease behind.
            CHECK_EQ(
                    queue.embedding(
                            table->view(), indices->view(), out->view(),
                            usable),
                    exhausted);
            // A disjoint aligned subrange of the same owner is independent.
            const iom::oid disjoint = queue.embedding(
                    table->view(), indices->view(), out->view(),
                    aligned.view().subrange(32, bytes));
            REQUIRE(iom::oid_is_token(disjoint));
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5220)),
                    std::size_t{1});
            queue.finish(token_sequence(leased));
            queue.finish(token_sequence(disjoint));
            CHECK_NOTHROW(queue.wait(leased));
            CHECK_NOTHROW(queue.wait(disjoint));
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5200)),
                    std::size_t{0});
            // Proven completion releases the range for reuse.
            const iom::oid reused = queue.embedding(
                    table->view(), indices->view(), out->view(), usable);
            REQUIRE(iom::oid_is_token(reused));
            queue.finish(token_sequence(reused));
            CHECK_NOTHROW(queue.wait(reused));
        }
    }

    // Queue order and accepted failures: a producer copy, the embedding, and an
    // output consumer are accepted in submission order; an out-of-vocabulary
    // index reaches the worker because admission never scans queued index data;
    // and an accepted data failure is a positive token whose waits rethrow
    // `std::invalid_argument` with an unspecified, never-compared output.
    {
        auto source = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto table = candidate.create_tensor(
                embedding_spec({3, 17}, iom::DataType::BF16));
        auto out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto destination = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        // Out-of-vocabulary and negative codes: a host admission scan would
        // reject them synchronously, so an accepted token is that proof.
        const std::function<std::uint64_t(std::size_t, std::size_t)>
                out_of_vocabulary = [](std::size_t, std::size_t row) {
                    return row == 0
                            ? std::uint64_t{0}
                            : std::numeric_limits<std::uint64_t>::max();
                };
        const std::vector<std::byte> index_bytes = encode_index_owner(
                embedding_spec({1, 17}, iom::DataType::U32), 17,
                out_of_vocabulary);
        ConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5700), 64);
        const EmbeddingCaseWindow window(observer);
        copy_from_host(indices->view(), index_bytes);
        CommonEmbeddingQueue queue(candidate, declaration.workspace);
        const iom::oid producer =
                queue.copy(source->view(), indices->view());
        REQUIRE(iom::oid_is_token(producer));
        const iom::oid embedded = declaration.workspace.bytes == 0
                ? queue.embedding(
                          table->view(), indices->view(), out->view())
                : queue.embedding(
                          table->view(), indices->view(), out->view(),
                          scratch.view().subrange(
                                  0, declaration.workspace.bytes));
        REQUIRE(iom::oid_is_token(embedded));
        const iom::oid consumer =
                queue.copy(out->view(), destination->view());
        REQUIRE(iom::oid_is_token(consumer));
        REQUIRE_EQ(queue.submissions().size(), std::size_t{3});
        CHECK_EQ(queue.submissions()[0], std::string_view{"copy"});
        CHECK_EQ(queue.submissions()[1], std::string_view{"embedding"});
        CHECK_EQ(queue.submissions()[2], std::string_view{"copy"});
        queue.finish(token_sequence(producer));
        queue.finish(token_sequence(embedded));
        queue.finish(token_sequence(consumer));
        CHECK_NOTHROW(queue.wait(producer));
        CHECK_NOTHROW(queue.wait(embedded));
        CHECK_NOTHROW(queue.wait(consumer));

        // The retained data failure: a positive token, the native completion
        // proof independent of the data error, and already accepted following
        // work allowed to execute while its dependent output is never consumed.
        queue.inject_failure(CommonEmbeddingQueue::Failure::post_acceptance);
        const iom::oid failed = declaration.workspace.bytes == 0
                ? queue.embedding(
                          table->view(), indices->view(), out->view())
                : queue.embedding(
                          table->view(), indices->view(), out->view(),
                          scratch.view().subrange(
                                  0, declaration.workspace.bytes));
        REQUIRE(iom::oid_is_token(failed));
        const iom::oid following =
                queue.copy(out->view(), destination->view());
        REQUIRE(iom::oid_is_token(following));
        queue.finish(token_sequence(failed));
        expect_repeated_invalid_argument(queue, failed);
        queue.finish(token_sequence(following));
        CHECK_NOTHROW(queue.wait(following));
    }

    // Unknown completion quarantines owners and the workspace lease; proven
    // completion releases the lease and keeps an accepted failure's owners
    // invalidated rather than reusable; a drained queue resets and the device
    // accepts a fresh queue with a safe valid operation and the same range.
    {
        auto table = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        auto indices = candidate.create_tensor(
                embedding_spec({1, 17}, iom::DataType::U32));
        auto out = candidate.create_tensor(
                embedding_spec({17, 17}, iom::DataType::BF16));
        ConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5800), 64);
        const EmbeddingCaseWindow window(observer);
        {
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            const iom::oid outstanding = declaration.workspace.bytes == 0
                    ? queue.embedding(
                              table->view(), indices->view(), out->view())
                    : queue.embedding(
                              table->view(), indices->view(), out->view(),
                              scratch.view().subrange(
                                      0, declaration.workspace.bytes));
            REQUIRE(iom::oid_is_token(outstanding));
            CHECK_EQ(
                    queue.registered_at(table->view().native_handle()),
                    std::size_t{1});
            if (declaration.workspace.bytes != 0) {
                // The lease is retained until completion is proven, so the
                // same range is not reusable yet.
                CHECK_EQ(
                        queue.registered_at(
                                reinterpret_cast<void*>(0x5800)),
                        std::size_t{1});
                CHECK_EQ(
                        queue.embedding(
                                table->view(), indices->view(), out->view(),
                                scratch.view().subrange(
                                        0, declaration.workspace.bytes)),
                        exhausted);
            }
            queue.finish(token_sequence(outstanding));
            CHECK_NOTHROW(queue.wait(outstanding));
            CHECK_EQ(
                    queue.registered_at(table->view().native_handle()),
                    std::size_t{0});
            if (declaration.workspace.bytes != 0) {
                CHECK_EQ(
                        queue.registered_at(
                                reinterpret_cast<void*>(0x5800)),
                        std::size_t{0});
            }
        }
        {
            // An accepted failure keeps its owners invalidated through the
            // quarantine even after its completion is proven, while the
            // workspace lease stays independent of the data error.
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            queue.inject_failure(
                    CommonEmbeddingQueue::Failure::post_acceptance);
            const iom::oid failed = declaration.workspace.bytes == 0
                    ? queue.embedding(
                              table->view(), indices->view(), out->view())
                    : queue.embedding(
                              table->view(), indices->view(), out->view(),
                              scratch.view().subrange(
                                      0, declaration.workspace.bytes));
            REQUIRE(iom::oid_is_token(failed));
            queue.finish(token_sequence(failed));
            expect_repeated_invalid_argument(queue, failed);
            CHECK_EQ(
                    queue.registered_at(table->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(indices->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(out->view().native_handle()),
                    std::size_t{1});
        }
        // After the drain, a safe valid operation and the same range prove
        // reset and release.
        {
            CommonEmbeddingQueue queue(candidate, declaration.workspace);
            const iom::oid token = declaration.workspace.bytes == 0
                    ? queue.embedding(
                              table->view(), indices->view(), out->view())
                    : queue.embedding(
                              table->view(), indices->view(), out->view(),
                              scratch.view().subrange(
                                      0, declaration.workspace.bytes));
            REQUIRE(iom::oid_is_token(token));
            queue.finish(token_sequence(token));
            CHECK_NOTHROW(queue.wait(token));
            CHECK_EQ(token_sequence(queue.probe()), 2);
        }
    }
}

// The complete shared embedding suite: the independent reference self-check,
// the declared request cases, and the common admission, ownership, queue-order,
// and failure cases.
inline void run_embedding_conformance(
        const ConformanceDevices& devices,
        const EmbeddingDeclaration& declaration,
        ConformanceObserver* observer = nullptr,
        AcceleratorStorageOracle* oracle = nullptr) {
    // The declared target matrix is the driver's explicit input and drives the
    // shared fixtures; an empty matrix would silently test nothing.
    REQUIRE_FALSE(declaration.target_payloads.empty());
    REQUIRE_FALSE(declaration.target_ids.empty());
    REQUIRE(run_embedding_oracle_self_check());
    run_embedding_reference_conformance(devices, declaration, observer, oracle);
    run_embedding_common_conformance(devices, declaration, observer);
}

}  // namespace iom_conformance