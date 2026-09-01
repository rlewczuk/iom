#pragma once

// Backend-neutral conformance harness: copy and storage scenarios (change
// 0001-tensor-view / 06).
//
// Provides the transformed-view and copy case matrices together with the
// storage-and-transfer, asynchronous-copy, copy-error, and transfer-error
// scenarios. Every scenario seeds matching reference and candidate tensors
// with independently generated logical host bytes and compares readbacks
// bit-for-bit. All shared types, helpers, and the case-parameter structs
// come from backend_conformance_common.hpp.

#include "backend/backend_conformance_common.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
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
// Transformed-view case matrix: final-dimension padding, leading ranks up to
// and above four, stepped slices, selects, every leading permutation when
// enumerable, contiguous reshapes, nested transforms, and rank-two
// boundaries. Builders only use public TensorView transforms.
// ---------------------------------------------------------------------------

struct ViewCase {
    std::string label;
    std::function<iom::TensorView(const iom::TensorView&)> build;
};

inline std::vector<ViewCase> view_cases_for(const iom::TensorSpec& owner_spec) {
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t leading = dims.size() - 2;

    std::vector<ViewCase> cases;
    cases.push_back({"full", [](const iom::TensorView& full) { return full; }});

    if (leading >= 1 && dims[0] >= 4) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"stepped slice of the first leading axis",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 / 2, 2);
                 }});
    }
    if (leading >= 1 && dims[0] >= 2) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"interior dense slice of the first leading axis",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 - 1);
                 }});
    }
    if (leading >= 1) {
        const std::size_t last_axis = leading - 1;
        const std::size_t last_dim = dims[last_axis];
        cases.push_back(
                {"select of the first leading axis",
                 [](const iom::TensorView& full) { return full.select(0, 0); }});
        cases.push_back(
                {"select of the last leading axis at its last index",
                 [last_axis, last_dim](const iom::TensorView& full) {
                     return full.select(last_axis, last_dim - 1);
                 }});
    }

    if (leading == 0) {
        cases.push_back(
                {"empty permute",
                 [](const iom::TensorView& full) {
                     const std::vector<std::size_t> empty;
                     return full.permute(std::span<const std::size_t>{empty});
                 }});
    } else if (leading <= 4) {
        std::vector<std::size_t> order(leading);
        std::iota(order.begin(), order.end(), std::size_t{0});
        do {
            const std::vector<std::size_t> captured = order;
            std::string label = "permute";
            for (const std::size_t axis : captured) {
                label += ' ';
                label += std::to_string(axis);
            }
            cases.push_back(
                    {std::move(label),
                     [captured](const iom::TensorView& full) {
                         return full.permute(
                                 std::span<const std::size_t>{captured});
                     }});
        } while (std::next_permutation(order.begin(), order.end()));
    }

    if (leading >= 1) {
        std::size_t product = 1;
        for (std::size_t i = 0; i < leading; ++i) {
            product *= dims[i];
        }
        cases.push_back(
                {"reshape to one leading axis",
                 [product](const iom::TensorView& full) {
                     return full.reshape_leading(span_of({product}));
                 }});
    }
    if (leading >= 2) {
        std::size_t head = 1;
        for (std::size_t i = 0; i < leading - 1; ++i) {
            head *= dims[i];
        }
        const std::size_t last_dim = dims[leading - 1];
        cases.push_back(
                {"merge the trailing pair of leading axes",
                 [head, last_dim](const iom::TensorView& full) {
                     return full.reshape_leading(span_of({head, last_dim}));
                 }});
    }
    if (leading >= 1 && dims[0] >= 4 && dims[0] % 2 == 0) {
        std::vector<std::size_t> split{2, dims[0] / 2};
        for (std::size_t i = 1; i < leading; ++i) {
            split.push_back(dims[i]);
        }
        cases.push_back(
                {"split the first leading axis",
                 [split](const iom::TensorView& full) {
                     return full.reshape_leading(
                             std::span<const std::size_t>{split});
                 }});
    }

    if (leading >= 3 && dims[0] >= 2 && dims[1] >= 3) {
        cases.push_back(
                {"select, stepped slice, and permute",
                 [](const iom::TensorView& full) {
                     return full.select(0, 1).slice(0, 0, 2, 2).permute(
                             span_of({1, 0}));
                 }});
    }
    if (leading >= 3 && dims[0] >= 2 && dims[2] >= 4) {
        const std::size_t d1 = dims[1];
        const std::size_t d2 = dims[2];
        cases.push_back(
                {"dense slice, reshape, select, and stepped slice",
                 [d1, d2](const iom::TensorView& full) {
                     return full.slice(0, 0, 2)
                             .reshape_leading(span_of({2 * d1, d2}))
                             .select(0, 2 * d1 - 1)
                             .slice(0, 0, d2 / 2, 2);
                 }});
    }

    return cases;
}

inline const std::vector<std::vector<std::size_t>>& transfer_owner_shapes() {
    static const std::vector<std::vector<std::size_t>> shapes = {
        {16, 16},                   // rank-two tile boundary
        {1, 17},                    // vector representation, row padding
        {17, 33},                   // both final dimensions padded
        {4, 3, 17, 33},             // leading rank two
        {2, 3, 4, 17, 33},          // leading rank three
        {2, 2, 2, 3, 17, 33},       // rank six: leading rank above three
    };
    return shapes;
}

// ---------------------------------------------------------------------------
// Copy case matrix: full windows, different source/destination offsets and
// strides, permuted windows, nested transforms, and the identical-window
// no-op. Source and destination windows never overlap, so destination bytes
// are deterministic after each copy.
// ---------------------------------------------------------------------------

struct CopyCase {
    std::string label;
    std::function<iom::TensorView(const iom::TensorView&)> build_source;
    std::function<iom::TensorView(const iom::TensorView&)> build_destination;
};

inline std::vector<CopyCase> copy_cases_for(const iom::TensorSpec& owner_spec) {
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t leading = dims.size() - 2;

    std::vector<CopyCase> cases;
    cases.push_back(
            {"full to full",
             [](const iom::TensorView& full) { return full; },
             [](const iom::TensorView& full) { return full; }});
    cases.push_back(
            {"identical window is a waitable no-op",
             [](const iom::TensorView& full) { return full; },
             [](const iom::TensorView& full) { return full; }});

    if (leading >= 1 && dims[0] >= 2) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"offset source to earlier destination",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 - 1);
                 },
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 - 1);
                 }});
    }
    if (leading >= 1 && dims[0] >= 4) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"stepped source to stepped destination",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 / 2, 2);
                 },
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 / 2, 2);
                 }});
    }
    if (leading >= 2) {
        std::vector<std::size_t> reversed(leading);
        std::iota(reversed.rbegin(), reversed.rend(), std::size_t{0});
        cases.push_back(
                {"permuted source to permuted destination",
                 [reversed](const iom::TensorView& full) {
                     return full.permute(std::span<const std::size_t>{reversed});
                 },
                 [reversed](const iom::TensorView& full) {
                     return full.permute(std::span<const std::size_t>{reversed});
                 }});
    }
    if (leading >= 2 && dims[1] >= 3) {
        cases.push_back(
                {"selected and stepped source to selected destination",
                 [](const iom::TensorView& full) {
                     return full.select(0, 1).slice(0, 0, 2, 2);
                 },
                 [](const iom::TensorView& full) {
                     return full.select(0, 0).slice(0, 0, 2, 2);
                 }});
    }

    return cases;
}

inline const std::vector<std::vector<std::size_t>>& copy_owner_shapes() {
    static const std::vector<std::vector<std::size_t>> shapes = {
        {17, 33},                   // rank-two padded
        {2, 3, 16, 16},             // exact tiles
        {2, 3, 4, 17, 33},          // leading rank three with padding
        {2, 2, 2, 3, 17, 33},       // rank six
    };
    return shapes;
}

// ---------------------------------------------------------------------------
// Copy and storage scenarios.
// ---------------------------------------------------------------------------

// Storage and host transfers: matching reference and candidate tensors are
// seeded with identical logical host bytes, full and transformed views
// transfer independently generated encodings on both devices, and every
// readback is compared bit-for-bit against the expected encoding.
inline void run_storage_and_transfer_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        for (const std::vector<std::size_t>& dimensions :
             transfer_owner_shapes()) {
            CAPTURE(dimensions);
            const iom::TensorSpec spec{iom::TensorShape{dimensions}, type};

            auto reference = devices.reference.create_tensor(spec);
            auto candidate = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            const std::vector<std::byte> seeded = encode_logical(spec, 0xABCD);
            reference->view().copy_from_host(seeded);
            candidate->view().copy_from_host(seeded);
            require_logical_bytes(
                    reference->view(), seeded, "reference full view");
            require_logical_bytes(
                    candidate->view(), seeded, "candidate full view");

            std::uint64_t salt = 1;
            for (const ViewCase& view_case : view_cases_for(spec)) {
                CAPTURE(view_case.label);
                iom::TensorView reference_view =
                        view_case.build(reference->view());
                iom::TensorView candidate_view =
                        view_case.build(candidate->view());
                REQUIRE(reference_view.spec() == candidate_view.spec());

                const std::vector<std::byte> pattern =
                        encode_logical(reference_view.spec(), salt);
                ++salt;
                reference_view.copy_from_host(pattern);
                candidate_view.copy_from_host(pattern);
                require_logical_bytes(
                        reference_view, pattern,
                        std::string("reference ") + view_case.label);
                require_logical_bytes(
                        candidate_view, pattern,
                        std::string("candidate ") + view_case.label);
            }

            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

// Asynchronous copies: every supported type exercises the in-order
// submission chain and full, offset, stepped, permuted, nested, and
// identical-window copies on both devices; destination logical bytes are
// compared bit-for-bit against the independently generated source encoding.
inline void run_async_copy_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));

        // Same-queue submissions execute in call order without intervening
        // host waits, and waits are idempotent after success.
        {
            const iom::TensorSpec spec{
                    iom::TensorShape{{2, 3, 16, 16}}, type};
            auto first = devices.candidate.create_tensor(spec);
            auto second = devices.candidate.create_tensor(spec);
            auto staging = devices.candidate.create_tensor(spec);
            auto destination = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            const std::vector<std::byte> first_pattern =
                    encode_logical(spec, 11);
            const std::vector<std::byte> second_pattern =
                    encode_logical(spec, 12);
            first->view().copy_from_host(first_pattern);
            second->view().copy_from_host(second_pattern);

            auto queue = devices.candidate.create_ops();
            const iom::oid one =
                    queue->copy(first->view(), staging->view());
            const iom::oid two =
                    queue->copy(staging->view(), destination->view());
            const iom::oid three =
                    queue->copy(second->view(), destination->view());
            CHECK_NE(token_queue(one), 0);
            CHECK_EQ(token_sequence(one), 1);
            CHECK_EQ(token_sequence(two), 2);
            CHECK_EQ(token_sequence(three), 3);
            CHECK_EQ(token_queue(one), token_queue(three));

            queue->wait(three);
            require_logical_bytes(
                    destination->view(), second_pattern,
                    "candidate chained destination");
            require_logical_bytes(
                    staging->view(), first_pattern, "candidate staging");
            queue->wait(one);
            queue->wait(three);
            queue.reset();

            if (observer != nullptr) {
                observer->case_complete();
            }
        }

        for (const std::vector<std::size_t>& dimensions : copy_owner_shapes()) {
            CAPTURE(dimensions);
            const iom::TensorSpec spec{iom::TensorShape{dimensions}, type};
            auto reference_source = devices.reference.create_tensor(spec);
            auto reference_destination =
                    devices.reference.create_tensor(spec);
            auto candidate_source = devices.candidate.create_tensor(spec);
            auto candidate_destination = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            auto reference_queue = devices.reference.create_ops();
            auto candidate_queue = devices.candidate.create_ops();

            std::uint64_t salt = 100;
            for (const CopyCase& copy_case : copy_cases_for(spec)) {
                CAPTURE(copy_case.label);
                iom::TensorView reference_source_view =
                        copy_case.build_source(reference_source->view());
                iom::TensorView reference_destination_view =
                        copy_case.build_destination(reference_destination->view());
                iom::TensorView candidate_source_view =
                        copy_case.build_source(candidate_source->view());
                iom::TensorView candidate_destination_view =
                        copy_case.build_destination(candidate_destination->view());
                REQUIRE(candidate_source_view.spec()
                        == candidate_destination_view.spec());

                const std::vector<std::byte> pattern =
                        encode_logical(candidate_source_view.spec(), salt);
                ++salt;
                reference_source_view.copy_from_host(pattern);
                candidate_source_view.copy_from_host(pattern);

                const iom::oid reference_token = reference_queue->copy(
                        reference_source_view, reference_destination_view);
                const iom::oid candidate_token = candidate_queue->copy(
                        candidate_source_view, candidate_destination_view);
                reference_queue->wait(reference_token);
                candidate_queue->wait(candidate_token);

                require_logical_bytes(
                        reference_destination_view, pattern,
                        std::string("reference ") + copy_case.label);
                require_logical_bytes(
                        candidate_destination_view, pattern,
                        std::string("candidate ") + copy_case.label);
            }

            // Every successful submission was waited; queues die explicitly.
            reference_queue.reset();
            candidate_queue.reset();
            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

// Copy validation: metadata mismatch and foreign-device views are rejected
// before any write and before consuming a sequence number.
inline void run_copy_error_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        const iom::DataType other_type =
                type == iom::DataType::F32 ? iom::DataType::I16
                                           : iom::DataType::F32;
        const iom::TensorSpec spec{iom::TensorShape{{2, 3, 17, 33}}, type};

        auto source = devices.candidate.create_tensor(spec);
        auto destination = devices.candidate.create_tensor(spec);
        auto other_shape = devices.candidate.create_tensor(
                iom::TensorSpec{iom::TensorShape{{3, 3, 17, 33}}, type});
        auto other_type_tensor = devices.candidate.create_tensor(
                iom::TensorSpec{iom::TensorShape{{2, 3, 17, 33}}, other_type});
        auto reference_view_tensor = devices.reference.create_tensor(spec);
        auto foreign_tensor = devices.foreign.create_tensor(spec);
        if (observer != nullptr) {
            observer->setup_complete();
        }

        auto queue = devices.candidate.create_ops();

        const std::vector<std::byte> source_pattern =
                encode_logical(spec, 21);
        const std::vector<std::byte> destination_pattern =
                encode_logical(spec, 22);
        source->view().copy_from_host(source_pattern);
        destination->view().copy_from_host(destination_pattern);

        CHECK_THROWS_AS(
                queue->copy(other_shape->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), other_shape->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(other_type_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), other_type_tensor->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(reference_view_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), reference_view_tensor->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(foreign_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), foreign_tensor->view()),
                std::invalid_argument);

        // No rejected submission wrote or consumed a sequence.
        require_logical_bytes(
                destination->view(), destination_pattern,
                "destination after rejections");
        const iom::oid valid = queue->copy(source->view(), destination->view());
        CHECK_EQ(token_sequence(valid), 1);
        queue->wait(valid);
        require_logical_bytes(
                destination->view(), source_pattern,
                "destination after the valid copy");

        // An identical-window copy still submits a waitable no-op.
        const iom::oid no_op =
                queue->copy(destination->view(), destination->view());
        CHECK_EQ(token_sequence(no_op), 2);
        queue->wait(no_op);
        require_logical_bytes(
                destination->view(), source_pattern,
                "destination after the no-op copy");
        queue.reset();

        if (observer != nullptr) {
            observer->case_complete();
        }
    }
}

// Synchronous transfer failures: wrong span sizes and non-canonical BOOL
// bytes are rejected, and metadata, owner identity, and the native handle
// are unchanged afterwards even though destination values may be
// unspecified.
inline void run_transfer_error_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        const iom::TensorSpec spec{iom::TensorShape{{2, 3, 17, 33}}, type};
        for (iom::Device* device :
             {&devices.reference, &devices.candidate}) {
            auto tensor = device->create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }
            iom::TensorView& view = tensor->view();

            const std::vector<std::byte> seeded =
                    encode_logical(spec, static_cast<std::uint64_t>(31));
            view.copy_from_host(seeded);

            const iom::TensorSpec spec_before = view.spec();
            const std::size_t offset_before = view.plane_offset();
            const std::vector<std::size_t> strides_before(
                    view.plane_strides().begin(), view.plane_strides().end());
            const iom::Device* device_before = &view.device();
            const void* handle_before = view.native_handle();

            std::vector<std::byte> short_write(seeded);
            short_write.pop_back();
            CHECK_THROWS_AS(view.copy_from_host(short_write),
                            std::invalid_argument);
            std::vector<std::byte> long_write(seeded);
            long_write.push_back(std::byte{0});
            CHECK_THROWS_AS(view.copy_from_host(long_write),
                            std::invalid_argument);

            std::vector<std::byte> short_read(
                    spec.logical_nbytes() - 1, kReadbackSentinel);
            CHECK_THROWS_AS(view.copy_to_host(short_read),
                            std::invalid_argument);
            CHECK(std::all_of(
                    short_read.begin(), short_read.end(),
                    [](std::byte value) { return value == kReadbackSentinel; }));
            std::vector<std::byte> long_read(
                    spec.logical_nbytes() + 1, kReadbackSentinel);
            CHECK_THROWS_AS(view.copy_to_host(long_read),
                            std::invalid_argument);
            CHECK(std::all_of(
                    long_read.begin(), long_read.end(),
                    [](std::byte value) { return value == kReadbackSentinel; }));

            if (type == iom::DataType::BOOL) {
                for (const std::size_t position :
                     {std::size_t{0}, seeded.size() / 2, seeded.size() - 1}) {
                    std::vector<std::byte> invalid = seeded;
                    invalid[position] = std::byte{2};
                    CAPTURE(position);
                    CHECK_THROWS_AS(view.copy_from_host(invalid),
                                    std::invalid_argument);
                }
                // Canonical zero and one bytes round-trip; non-canonical
                // bytes never wrote.
                require_logical_bytes(
                        view, seeded, "storage untouched after invalid BOOL");
            }

            // Metadata, owner identity, and native handle survive every
            // synchronous failure; values are deliberately not compared.
            CHECK(view.spec() == spec_before);
            CHECK_EQ(view.plane_offset(), offset_before);
            CHECK(std::equal(
                    strides_before.begin(), strides_before.end(),
                    view.plane_strides().begin(), view.plane_strides().end()));
            CHECK_EQ(device_before, &view.device());
            CHECK_EQ(handle_before, view.native_handle());

            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

}  // namespace iom_conformance
