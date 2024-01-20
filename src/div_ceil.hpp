#ifndef ROOFBENCH_DIV_CEIL_HPP
#define ROOFBENCH_DIV_CEIL_HPP

#include <concepts>

namespace roofbench {

template <std::unsigned_integral T>
static inline T div_ceil(const T &a, const T &b) noexcept {
    return a % b != 0 ? a / b + 1 : a / b;
}

} // namespace roofbench

#endif
