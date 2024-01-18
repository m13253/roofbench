#include "perf_timer.hpp"
#include <cerrno>
#include <system_error>
#include <utility>

namespace roofbench {

PerfDuration &PerfDuration::operator+=(const PerfDuration &other) noexcept {
    duration.tv_sec += other.duration.tv_sec;
    duration.tv_nsec += other.duration.tv_nsec;
    if (duration.tv_nsec >= 1'000'000'000) {
        duration.tv_sec++;
        duration.tv_nsec -= 1'000'000'000;
    }
    return *this;
}

std::chrono::duration<double> PerfDuration::seconds() const noexcept {
    return std::chrono::duration<double>(duration.tv_sec + duration.tv_nsec * 1e-9);
}

PerfTimer PerfTimer::now() {
    std::timespec time;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &time) != 0) {
        throw std::system_error(errno, std::generic_category());
    }
    return PerfTimer(std::move(time));
}

PerfDuration PerfTimer::operator-(const PerfTimer &since) const noexcept {
    auto tv_sec = time.tv_sec - since.time.tv_sec;
    auto tv_nsec = time.tv_nsec - since.time.tv_nsec;
    if (time.tv_nsec < since.time.tv_nsec) {
        tv_sec--;
        tv_nsec += 1'000'000'000;
    }
    return PerfDuration(std::timespec{
        .tv_sec = tv_sec,
        .tv_nsec = tv_nsec,
    });
}

} // namespace roofbench
