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
    app.add_option("-f,--num-f32-ops"s, options.num_float_ops_f32, fmt::format("Number of floating-point operations per thread [Default: {}]", options.num_float_ops_f32));
    app.add_option("-d,--num-f64-ops"s, options.num_float_ops_f64, fmt::format("Number of floating-point operations per thread [Default: {}]", options.num_float_ops_f64));
    app.add_option("-s,--mem-read-size"s, options.mem_read_size, fmt::format("Memory read size per thread [Default: {}]"sv, options.mem_read_size));
    app.add_option("-m,--num-memory-reads"s, options.num_mem_reads, fmt::format("Number of memory reads [Default: {}]"sv, options.num_mem_reads));
    app.add_option("-l,--num-latency-measures"s, options.num_latency_measures, fmt::format("Number of inter-thread latency measurements [Default: {}]"sv, options.num_latency_measures));
    CLI11_PARSE(app, argc, argv);

    return roofbench::benchmark(options);
}
