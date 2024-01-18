#ifndef ROOFBENCH_PERF_TIMER_HPP
#define ROOFBENCH_PERF_TIMER_HPP

#include <chrono>
#include <ctime>
#include <fmt/core.h>

namespace roofbench {

class PerfDuration final {
public:
    explicit PerfDuration() noexcept = default;
    explicit PerfDuration(const std::timespec &duration) noexcept :
        duration(duration) {}
    PerfDuration &operator+=(const PerfDuration &other) noexcept;
    std::chrono::duration<double> seconds() const noexcept;

    std::timespec duration = {};
};

class PerfTimer final {
public:
    explicit PerfTimer() noexcept = default;
    explicit PerfTimer(const std::timespec &time) noexcept :
        time(time) {}
    PerfDuration operator-(const PerfTimer &since) const noexcept;
    static PerfTimer now();

private:
    std::timespec time = {};
};

} // namespace roofbench

template <>
struct fmt::formatter<roofbench::PerfDuration> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const roofbench::PerfDuration &self, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}.{:09}", self.duration.tv_sec, self.duration.tv_nsec);
    }
};

#endif
