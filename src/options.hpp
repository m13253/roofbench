#ifndef ROOFBENCH_OPTIONS_HPP
#define ROOFBENCH_OPTIONS_HPP

#include <cstddef>
#include <cstdint>

namespace roofbench {

struct AppOptions {
    static constexpr size_t simd_batch_size = ROOFBENCH_SIMD_BATCH_SIZE;

    uint64_t num_float_ops = 274877906944;
    size_t mem_write_size = 134217728;
    size_t num_mem_writes = 64;
    size_t num_latency_measures = 2048;
};

} // namespace roofbench

#endif
