#pragma once

#include <cstddef>

#ifdef IOM_ENABLE_TESTING
namespace iom::ttnn_detail {

void consume_quarantine_action_fault_locked() noexcept(false);
void consume_copy_registration_fault() noexcept(false);
void consume_copy_outcome_insertion_fault() noexcept(false);
void consume_binary_outcome_insertion_fault() noexcept(false);
[[nodiscard]] bool consume_copy_finish_fault() noexcept;
void fail_copy_planes_submission_at(std::size_t index) noexcept(false);
void fail_host_transfer_submission_at(std::size_t index) noexcept(false);

}  // namespace iom::ttnn_detail
#endif
