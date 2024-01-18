#ifndef ROOFBENCH_PAUSE_HPP
#define ROOFBENCH_PAUSE_HPP

namespace roofbench {

static inline void ia32_pause() noexcept {
#if defined(__amd64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

} // namespace roofbench

#endif
