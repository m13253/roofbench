#ifndef ROOFBENCH_SPIN_BARRIER_HPP
#define ROOFBENCH_SPIN_BARRIER_HPP

#include "pause.hpp"
#include <atomic>

namespace roofbench {

class SpinBarrier final {
public:
    explicit inline SpinBarrier(std::size_t count = 0) noexcept :
        spinner(count) {
    }
    inline void init(std::size_t count) noexcept {
        spinner.store(count, std::memory_order_relaxed);
    }
    inline void wait() noexcept {
        if (spinner.fetch_sub(1, std::memory_order_release) != 1) {
            do {
                ia32_pause();
            } while (spinner.load(std::memory_order_acquire) != 0);
        }
    }

private:
    std::atomic_size_t spinner;
};

} // namespace roofbench

#endif
