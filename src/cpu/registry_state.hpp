#pragma once

#include "iom/detail/outstanding_work_registry.hpp"

namespace iom::cpu_detail {

struct CpuRegistryState {
    detail::OutstandingWorkRegistry registry;
    detail::Quarantine quarantine;
};

}  // namespace iom::cpu_detail
