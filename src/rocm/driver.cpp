#include "driver.hpp"

namespace iom::rocm_detail {

#ifdef IOM_ENABLE_TESTING
AllocationObserver allocation_observer{};
AllocationCalls allocation_calls{};
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::rocm_detail