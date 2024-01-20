#ifndef ROOFBENCH_PERF_TIMER_HPP
#define ROOFBENCH_PERF_TIMER_HPP

#include <cerrno>
#include <chrono>
#include <ctime>
#include <fmt/core.h>
#include <system_error>
#include <utility>

namespace roofbench {

class PerfDuration final {
public:
    explicit PerfDuration() noexcept = default;
    explicit PerfDuration(const std::timespec &duration) noexcept :
        duration(duration) {}
    inline PerfDuration &operator+=(const PerfDuration &other) noexcept {
        duration.tv_sec += other.duration.tv_sec;
        duration.tv_nsec += other.duration.tv_nsec;
        if (duration.tv_nsec >= 1'000'000'000) {
            duration.tv_sec++;
            duration.tv_nsec -= 1'000'000'000;
        }
        return *this;
    }
    inline std::chrono::duration<double> seconds() const noexcept {
        return std::chrono::duration<double>(duration.tv_sec + duration.tv_nsec * 1e-9);
    }

    std::timespec duration = {};
};

class PerfTimer final {
public:
    explicit PerfTimer() noexcept = default;
    explicit PerfTimer(const std::timespec &time) noexcept :
        time(time) {}
    inline PerfDuration operator-(const PerfTimer &since) const noexcept {
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
    static inline PerfTimer now() {
        std::timespec time;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &time) != 0) {
            throw std::system_error(errno, std::generic_category());
        }
        return PerfTimer(std::move(time));
    }

private:
    std::timespec time = {};
};

} // namespace roofbench

template <>
struct fmt::formatter<roofbench::PerfDuration> {
    template <typename ParseContext>
    inline constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    inline auto format(const roofbench::PerfDuration &self, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}.{:09}", self.duration.tv_sec, self.duration.tv_nsec);
    }
};

#endif
