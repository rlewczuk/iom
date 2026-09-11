#pragma once

// Backend-neutral memory, admission, rank, and workspace contract. No backend
// kind or runtime header is used here.

#include "backend/backend_conformance_common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <set>
#include <stdexcept>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

inline void run_queue_capacity_conformance(
        const ConformanceDevices& devices) {
    CHECK_EQ(
            devices.candidate.queue_config().max_in_flight_per_queue,
            std::size_t{16});

    std::array<std::unique_ptr<iom::DeviceOps>, 4> candidate_queues;
    std::array<std::unique_ptr<iom::DeviceOps>, 4> foreign_queues;
    for (std::size_t index = 0; index < candidate_queues.size(); ++index) {
        candidate_queues[index] = devices.candidate.create_ops();
        foreign_queues[index] = devices.foreign.create_ops();
        REQUIRE(candidate_queues[index] != nullptr);
        REQUIRE(foreign_queues[index] != nullptr);
    }
    CHECK_THROWS_AS((void)devices.candidate.create_ops(), std::bad_alloc);
    CHECK_THROWS_AS((void)devices.foreign.create_ops(), std::bad_alloc);

    // A failed fifth creation must not consume the released capacity.
    candidate_queues[0].reset();
    foreign_queues[0].reset();
    candidate_queues[0] = devices.candidate.create_ops();
    foreign_queues[0] = devices.foreign.create_ops();
    REQUIRE(candidate_queues[0] != nullptr);
    REQUIRE(foreign_queues[0] != nullptr);

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto source = devices.candidate.create_tensor(spec);
    std::array<std::unique_ptr<iom::Tensor>, 4> destinations;
    for (auto& destination : destinations) {
        destination = devices.candidate.create_tensor(spec);
    }
    copy_from_host(source->view(), encode_logical(spec, 0xA11CE));

    std::set<std::uint8_t> queue_ids;
    std::array<iom::oid, 4> tokens{};
    for (std::size_t index = 0; index < candidate_queues.size(); ++index) {
        tokens[index] = candidate_queues[index]->copy(
                source->view(), destinations[index]->view());
        REQUIRE(iom::oid_is_token(tokens[index]));
        queue_ids.insert(token_queue(tokens[index]));
    }
    CHECK_EQ(queue_ids.size(), candidate_queues.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        CHECK_NOTHROW(candidate_queues[index]->wait(tokens[index]));
        CHECK_NOTHROW(candidate_queues[index]->wait(tokens[index]));
    }
}

inline void run_workspace_contract_conformance(
        const ConformanceDevices& devices) {
    const auto supported = devices.candidate.supported_data_types();
    const auto f32 = std::find(
            supported.begin(), supported.end(), iom::DataType::F32);
    REQUIRE(f32 != supported.end());

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    auto source = devices.candidate.create_tensor(spec);
    auto destination = devices.candidate.create_tensor(spec);
    const std::vector<std::byte> input = encode_logical(spec, 0x5100);

    const iom::WorkspaceRequirements from_requirements =
            source->view().copy_from_host_workspace_requirements();
    CHECK_EQ(
            from_requirements,
            source->view().copy_from_host_workspace_requirements());
    const iom::WorkspaceRequirements to_requirements =
            destination->view().copy_to_host_workspace_requirements();
    CHECK_EQ(
            to_requirements,
            destination->view().copy_to_host_workspace_requirements());

    if (from_requirements.bytes == 0) {
        source->view().copy_from_host(input);
    } else {
        CHECK_THROWS_AS(source->view().copy_from_host(input),
                        std::invalid_argument);
        auto workspace = devices.candidate.create_workspace(
                from_requirements.bytes);
        REQUIRE(workspace != nullptr);
        source->view().copy_from_host(input, workspace->view());

        // A range from another exact Device is never a valid borrowed
        // workspace, even when the byte count and backend ordinal match.
        auto foreign_workspace = devices.foreign.create_workspace(
                from_requirements.bytes);
        REQUIRE(foreign_workspace != nullptr);
        CHECK_THROWS_AS(
                source->view().copy_from_host(
                        input, foreign_workspace->view()),
                std::invalid_argument);

        // The live-owner identity is part of the checked view contract.
        auto stale_workspace = devices.candidate.create_workspace(
                from_requirements.bytes);
        REQUIRE(stale_workspace != nullptr);
        const iom::RawWorkspaceView stale_view = stale_workspace->view();
        stale_workspace.reset();
        CHECK_THROWS_AS(
                source->view().copy_from_host(input, stale_view),
                std::invalid_argument);

        if (from_requirements.bytes > 1) {
            auto undersized = devices.candidate.create_workspace(
                    from_requirements.bytes - 1);
            REQUIRE(undersized != nullptr);
            CHECK_THROWS_AS(
                    source->view().copy_from_host(input, undersized->view()),
                    std::invalid_argument);
            CHECK_THROWS_AS(
                    (void)undersized->view().subrange(1, 1),
                    std::invalid_argument);
        }
    }

    std::vector<std::byte> output(spec.logical_nbytes(), kReadbackSentinel);
    if (to_requirements.bytes == 0) {
        destination->view().copy_to_host(output);
    } else {
        auto workspace = devices.candidate.create_workspace(
                to_requirements.bytes);
        REQUIRE(workspace != nullptr);
        destination->view().copy_to_host(output, workspace->view());
    }

    auto lhs = devices.candidate.create_tensor(spec);
    auto rhs = devices.candidate.create_tensor(spec);
    auto result = devices.candidate.create_tensor(spec);
    copy_from_host(lhs->view(), input);
    copy_from_host(rhs->view(), encode_logical(spec, 0x5200));
    copy_from_host(
            result->view(),
            std::vector<std::byte>(spec.logical_nbytes(), std::byte{0}));

    auto queue = devices.candidate.create_ops();
    for (const BinaryOperation operation : {
                 BinaryOperation::add, BinaryOperation::mul,
                 BinaryOperation::sub, BinaryOperation::div}) {
        const iom::WorkspaceRequirements requirements =
                query_binary_workspace_requirements(
                        *queue, operation, lhs->view(), rhs->view(),
                        result->view());
        CHECK_EQ(
                requirements,
                query_binary_workspace_requirements(
                        *queue, operation, lhs->view(), rhs->view(),
                        result->view()));

        if (requirements.bytes == 0) {
            const iom::oid token = submit_binary_operation(
                    *queue, operation, lhs->view(), rhs->view(),
                    result->view());
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
            continue;
        }

        CHECK_EQ(
                submit_binary_operation(
                        *queue, operation, lhs->view(), rhs->view(),
                        result->view()),
                iom::to_oid(iom::OidError::InvalidArgument));

        auto undersized = devices.candidate.create_workspace(
                requirements.bytes - 1);
        REQUIRE(undersized != nullptr);
        CHECK_EQ(
                submit_binary_operation(
                        *queue, operation, lhs->view(), rhs->view(),
                        result->view(), undersized->view()),
                iom::to_oid(iom::OidError::InvalidArgument));

        const std::size_t aligned_offset =
                ((requirements.bytes + 31) / 32) * 32;
        const std::size_t disjoint_capacity =
                aligned_offset + requirements.bytes;
        auto disjoint = devices.candidate.create_workspace(disjoint_capacity);
        REQUIRE(disjoint != nullptr);
        const iom::RawWorkspaceView first =
                disjoint->view().subrange(0, requirements.bytes);
        const iom::RawWorkspaceView second = disjoint->view().subrange(
                aligned_offset, requirements.bytes);
        const iom::oid first_token = submit_binary_operation(
                *queue, operation, lhs->view(), rhs->view(), result->view(),
                first);
        const iom::oid second_token = submit_binary_operation(
                *queue, operation, lhs->view(), rhs->view(), result->view(),
                second);
        REQUIRE(iom::oid_is_token(first_token));
        REQUIRE(iom::oid_is_token(second_token));
        CHECK_NOTHROW(queue->wait(first_token));
        CHECK_NOTHROW(queue->wait(second_token));
    }
}

// This composes the final cross-backend memory contract. Native setup counts,
// backing addresses, allocator fragmentation, and backend fault seams remain
// in each driver's smoke/conformance source; every public memory boundary is
// exercised here without duplicating backend-specific implementation logic.
inline void run_memory_contract_conformance(
        const ConformanceDevices& devices) {
    run_accelerator_rank_boundary_conformance(devices.candidate);
    run_queue_capacity_conformance(devices);
    run_workspace_contract_conformance(devices);
}

}  // namespace iom_conformance

