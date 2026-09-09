#include "copy.hpp"

// ADD execution is kept in copy.cpp so it can share the queue's private
// in-order submission and fence machinery. This translation unit is retained
// as the backend-private ADD compilation boundary for future device kernels.
