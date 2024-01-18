#ifndef ROOFBENCH_DENORM_HPP
#define ROOFBENCH_DENORM_HPP

#include <cstdint>

namespace roofbench {

static inline void enable_denorm_ftz() noexcept {
#if defined(__amd64__) || defined(__i386__)
    uint32_t mxcsr;
    asm("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x8040;
    asm("ldmxcsr %0" : : "m"(mxcsr));
#endif
#if defined(__aarch64__) || defined(__arm64__)
    uintptr_t fpcr;
    asm("mrs %0, fpcr" : "=r"(fpcr));
    asm("msr fpcr, %0" : : "r"(fpcr | 0x1000000));
#endif
}

} // namespace roofbench

#endif
