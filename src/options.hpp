#ifndef ROOFBENCH_OPTIONS_HPP
#define ROOFBENCH_OPTIONS_HPP

#include <cstddef>
#include <cstdint>

namespace roofbench {

template <typename T>
struct FloatBatchSize;

struct AppOptions final {
    std::uint64_t num_float_ops_f32 = 274877906944;
    std::uint64_t num_float_ops_f64 = 137438953472;
    std::size_t mem_read_size = 134217728;
    std::size_t num_mem_reads = 64;
    std::size_t num_latency_measures = 2048;

    template <typename T>
    static constexpr std::size_t float_batch_size = FloatBatchSize<T>::float_batch_size;
};

template<>
struct FloatBatchSize<float> final {
    static constexpr std::size_t float_batch_size = ROOFBENCH_SIMD_BATCH_SIZE_F32;
};
template<>
struct FloatBatchSize<double> final {
    static constexpr std::size_t float_batch_size = ROOFBENCH_SIMD_BATCH_SIZE_F64;
};

} // namespace roofbench

#endif
