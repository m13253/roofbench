#include "bench.hpp"
#include "options.hpp"
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <string>
#include <string_view>

using namespace std::literals;

int main(int argc, char *argv[]) {
    roofbench::AppOptions options;

    CLI::App app{"Roofbench"s};
    app.add_option("-a,--num-float-add"s, options.num_float_add, fmt::format("Number of floating-point add operations per thread [Default: {}]", options.num_float_add));
    app.add_option("-m,--num-float-mul"s, options.num_float_mul, fmt::format("Number of floating-point multiply operations per thread [Default: {}]", options.num_float_mul));
    app.add_option("-s,--mem-write-size"s, options.mem_write_size, fmt::format("Memory write size per thread [Default: {}]"sv, options.mem_write_size));
    app.add_option("-w,--num-memory-writes"s, options.num_mem_writes, fmt::format("Number of memory writes [Default: {}]"sv, options.num_mem_writes));
    app.add_option("-l,--num-latency-measures"s, options.num_latency_measures, fmt::format("Number of inter-thread latency measurements [Default: {}]"sv, options.num_latency_measures));
    CLI11_PARSE(app, argc, argv);

    return roofbench::benchmark(options);
}
