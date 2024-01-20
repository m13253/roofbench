#include "bench.hpp"
#include "denorm.hpp"
#include "div_ceil.hpp"
#include "options.hpp"
#include "pause.hpp"
#include "perf_timer.hpp"
#include "spin_barrier.hpp"
#include <array>
#include <atomic>
#include <benchmark/benchmark.h>
#include <cerrno>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fmt/core.h>
#include <limits>
#include <memory>
#include <new>
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <semaphore>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

using namespace std::literals;

namespace roofbench {

struct BenchStorage final {
    int omp_affinity;
    unsigned kernel_affinity;
    unsigned numa_node;

    uint64_t num_float_batches_f32;
    uint64_t num_float_batches_f64;
    uint64_t num_float_add_ops_f32;
    uint64_t num_float_mul_ops_f32;
    uint64_t num_float_fma_ops_f32;
    uint64_t num_float_add_ops_f64;
    uint64_t num_float_mul_ops_f64;
    uint64_t num_float_fma_ops_f64;
    PerfDuration float_add_duration_f32;
    PerfDuration float_mul_duration_f32;
    PerfDuration float_fma_duration_f32;
    PerfDuration float_add_duration_f64;
    PerfDuration float_mul_duration_f64;
    PerfDuration float_fma_duration_f64;

    PerfDuration mem_write_duration;
    std::vector<PerfDuration> latency_duration;

    size_t pagesize;
    size_t mem_write_size;
    size_t num_mem_writes;
    size_t num_latency_measures;

    void *mem_write_buf;
    void *latency_flag_buf;

    std::atomic_bool *latency_flag;
    std::binary_semaphore latency_host_sem{0};
    std::binary_semaphore latency_guest_sem{0};

    explicit BenchStorage(const AppOptions &options, int num_threads) {
        omp_affinity = omp_get_place_num();
        if (getcpu(&kernel_affinity, &numa_node) != 0) {
            throw std::system_error(errno, std::generic_category());
        }

        num_float_batches_f32 = div_ceil<uint64_t>(options.num_float_ops_f32, AppOptions::float_batch_size<float> * (uint64_t) 2);
        num_float_batches_f64 = div_ceil<uint64_t>(options.num_float_ops_f64, AppOptions::float_batch_size<double> * (uint64_t) 2);
        latency_duration = std::vector<PerfDuration>(num_threads);

        pagesize = numa_pagesize();
        mem_write_size = div_ceil(options.mem_write_size, pagesize) * pagesize;
        num_mem_writes = options.num_mem_writes;
        num_latency_measures = options.num_latency_measures;

        latency_flag_buf = numa_alloc_local(pagesize + mem_write_size);
        if (!latency_flag_buf) {
            throw std::bad_alloc();
        }
        mem_write_buf = &((std::byte *) latency_flag_buf)[pagesize];
        std::memset(latency_flag_buf, 0xcc, pagesize + mem_write_size);
        benchmark::DoNotOptimize(latency_flag_buf);

        latency_flag = new (latency_flag_buf) std::atomic_bool(false);
    }

    BenchStorage(const BenchStorage &) = delete;
    BenchStorage &operator=(const BenchStorage &) = delete;

    ~BenchStorage() noexcept {
        latency_flag->~atomic();
        numa_free(latency_flag_buf, pagesize + mem_write_size);
    }
};

template <std::floating_point T>
static inline uint64_t benchmark_float_add(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_add<float>(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_add<double>(uint64_t num_batches);
template <std::floating_point T>
static inline uint64_t benchmark_float_mul(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_mul<float>(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_mul<double>(uint64_t num_batches);
template <std::floating_point T>
static inline uint64_t benchmark_float_fma(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_fma<float>(uint64_t num_batches);
template <>
inline uint64_t benchmark_float_fma<double>(uint64_t num_batches);
static inline void benchmark_mem_write(const BenchStorage &local_storage);
static inline PerfDuration benchmark_latency_host(const BenchStorage &host_storage);
static inline void benchmark_latency_guest(const BenchStorage &host_storage);

static inline void print_affinity(std::span<const std::unique_ptr<BenchStorage>> storage);
static inline void print_flops(std::span<const std::unique_ptr<BenchStorage>> storage, const PerfDuration BenchStorage::*duration, const uint64_t BenchStorage::*num_float_ops, double &flops_sum);
static inline void print_mem_write(std::span<const std::unique_ptr<BenchStorage>> storage, double &throughput_sum);
static inline void print_latency(std::span<const std::unique_ptr<BenchStorage>> storage, double &latency_rtt_sum, double &latency_rtt_max);

int benchmark(const AppOptions &options) {
    int retval = 0;
    std::vector<std::unique_ptr<BenchStorage>> storage;
    double float_add_flops_sum_f32;
    double float_mul_flops_sum_f32;
    double float_fma_flops_sum_f32;
    double float_add_flops_sum_f64;
    double float_mul_flops_sum_f64;
    double float_fma_flops_sum_f64;
    double mem_write_throughput_sum;
    double latency_rtt_sum;
    double latency_rtt_max;
    std::array<SpinBarrier, 14> spin_barriers;

#pragma omp parallel
    {
#pragma omp master
        switch (omp_get_proc_bind()) {
        default:
            fmt::println(stderr, "Error: environment variable OMP_PROC_BIND should be true, close, or spread."sv);
            retval = 1;
        case omp_proc_bind_true:
        case omp_proc_bind_close:
        case omp_proc_bind_spread:;
        }
#pragma omp barrier
        if (retval == 0) {
            enable_denorm_ftz();
            numa_set_localalloc();
            numa_set_strict(true);
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
#pragma omp master
            {
                fmt::print("{{\n    \"affinity\": {{\n"sv);
                storage = std::vector<std::unique_ptr<BenchStorage>>(num_threads);
                for (auto &i : spin_barriers) {
                    i.init(num_threads);
                }
            }
#pragma omp barrier
            BenchStorage &local_storage = *(storage[thread_num] = std::make_unique<BenchStorage>(options, num_threads)).get();
#pragma omp barrier
#pragma omp master
            {
                print_affinity(storage);
                fmt::print("\n    }},\n    \"f32_add\": {{\n"sv);
            }
            spin_barriers[0].wait();
            PerfTimer start = PerfTimer::now();
            spin_barriers[1].wait();
            uint64_t num_float_ops = benchmark_float_add<float>(local_storage.num_float_batches_f32);
            PerfTimer finish = PerfTimer::now();
            local_storage.num_float_add_ops_f32 = num_float_ops;
            local_storage.float_add_duration_f32 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_add_duration_f32, &BenchStorage::num_float_add_ops_f32, float_add_flops_sum_f32);
                fmt::print("\n    }},\n    \"f32_mul\": {{\n"sv);
            }
            spin_barriers[2].wait();
            start = PerfTimer::now();
            spin_barriers[3].wait();
            num_float_ops = benchmark_float_mul<float>(local_storage.num_float_batches_f32);
            finish = PerfTimer::now();
            local_storage.num_float_mul_ops_f32 = num_float_ops;
            local_storage.float_mul_duration_f32 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_mul_duration_f32, &BenchStorage::num_float_mul_ops_f32, float_mul_flops_sum_f32);
                fmt::print("\n    }},\n    \"f32_fma\": {{\n"sv);
            }
            spin_barriers[4].wait();
            start = PerfTimer::now();
            spin_barriers[5].wait();
            num_float_ops = benchmark_float_fma<float>(local_storage.num_float_batches_f32);
            finish = PerfTimer::now();
            local_storage.num_float_fma_ops_f32 = num_float_ops;
            local_storage.float_fma_duration_f32 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_fma_duration_f32, &BenchStorage::num_float_fma_ops_f32, float_fma_flops_sum_f32);
                fmt::print("\n    }},\n    \"f64_add\": {{\n"sv);
            }
            spin_barriers[6].wait();
            start = PerfTimer::now();
            spin_barriers[7].wait();
            num_float_ops = benchmark_float_add<double>(local_storage.num_float_batches_f64);
            finish = PerfTimer::now();
            local_storage.num_float_add_ops_f64 = num_float_ops;
            local_storage.float_add_duration_f64 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_add_duration_f64, &BenchStorage::num_float_add_ops_f64, float_add_flops_sum_f64);
                fmt::print("\n    }},\n    \"f64_mul\": {{\n"sv);
            }
            spin_barriers[8].wait();
            start = PerfTimer::now();
            spin_barriers[9].wait();
            num_float_ops = benchmark_float_mul<double>(local_storage.num_float_batches_f64);
            finish = PerfTimer::now();
            local_storage.num_float_mul_ops_f64 = num_float_ops;
            local_storage.float_mul_duration_f64 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_mul_duration_f64, &BenchStorage::num_float_mul_ops_f64, float_mul_flops_sum_f64);
                fmt::print("\n    }},\n    \"f64_fma\": {{\n"sv);
            }
            spin_barriers[10].wait();
            start = PerfTimer::now();
            spin_barriers[11].wait();
            num_float_ops = benchmark_float_fma<double>(local_storage.num_float_batches_f64);
            finish = PerfTimer::now();
            local_storage.num_float_fma_ops_f64 = num_float_ops;
            local_storage.float_fma_duration_f64 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_fma_duration_f64, &BenchStorage::num_float_fma_ops_f64, float_fma_flops_sum_f64);
                fmt::print("\n    }},\n    \"mem_write\": {{\n"sv);
            }
            spin_barriers[12].wait();
            start = PerfTimer::now();
            spin_barriers[13].wait();
            benchmark_mem_write(local_storage);
            finish = PerfTimer::now();
            local_storage.mem_write_duration = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_mem_write(storage, mem_write_throughput_sum);
                fmt::print("\n    }},\n    \"inter_thread_latency\": {{\n"sv);
            }
            if (thread_num != 0) {
                for (int i = 0; i < thread_num; i++) {
                    local_storage.latency_guest_sem.acquire();
                    benchmark_latency_guest(*storage[i]);
                }
                local_storage.latency_host_sem.acquire();
            }
            for (int i = 0; i < thread_num; i++) {
                storage[i]->latency_guest_sem.release();
                local_storage.latency_duration[i] = benchmark_latency_host(local_storage);
            }
            for (int i = thread_num + 1; i < num_threads; i++) {
                storage[i]->latency_guest_sem.release();
                local_storage.latency_duration[i] = benchmark_latency_host(local_storage);
            }
            if (thread_num + 1 == num_threads) {
                storage[0]->latency_host_sem.release();
            } else {
                storage[thread_num + 1]->latency_host_sem.release();
                for (int i = thread_num + 1; i < num_threads; i++) {
                    local_storage.latency_guest_sem.acquire();
                    benchmark_latency_guest(*storage[i]);
                }
            }
#pragma omp master
            {
                local_storage.latency_host_sem.acquire();
                print_latency(storage, latency_rtt_sum, latency_rtt_max);
            }
#pragma omp barrier
            storage[thread_num].reset();
#pragma omp master
            {
                fmt::print("\n    }},\n    \"summary\": {{\n        \"flops_f32\": {{"sv);
                if (!std::isfinite(float_add_flops_sum_f32)) {
                    fmt::print("\"add\": null, "sv);
                } else {
                    fmt::print("\"add\": {:.16e}, "sv, float_add_flops_sum_f32);
                }
                if (!std::isfinite(float_mul_flops_sum_f32)) {
                    fmt::print("\"mul\": null, "sv);
                } else {
                    fmt::print("\"mul\": {:.16e}, "sv, float_mul_flops_sum_f32);
                }
                if (!std::isfinite(float_fma_flops_sum_f32)) {
                    fmt::print("\"fma\": null"sv);
                } else {
                    fmt::print("\"fma\": {:.16e}"sv, float_fma_flops_sum_f32);
                }
                fmt::print("}},\n        \"flops_f64\": {{"sv);
                if (!std::isfinite(float_add_flops_sum_f64)) {
                    fmt::print("\"add\": null, "sv);
                } else {
                    fmt::print("\"add\": {:.16e}, "sv, float_add_flops_sum_f64);
                }
                if (!std::isfinite(float_mul_flops_sum_f64)) {
                    fmt::print("\"mul\": null, "sv);
                } else {
                    fmt::print("\"mul\": {:.16e}, "sv, float_mul_flops_sum_f64);
                }
                if (!std::isfinite(float_fma_flops_sum_f64)) {
                    fmt::print("\"fma\": null"sv);
                } else {
                    fmt::print("\"fma\": {:.16e}"sv, float_fma_flops_sum_f64);
                }
                if (!std::isfinite(mem_write_throughput_sum)) {
                    fmt::print("}},\n        \"mem_write_throughput\": null,\n"sv);
                } else {
                    fmt::print("}},\n        \"mem_write_throughput\": {:.16e},\n"sv, mem_write_throughput_sum);
                }
                fmt::print("        \"inter_thread_latency_rtt\": {{");
                double latency_rtt_avg = latency_rtt_sum / (num_threads * num_threads);
                if (!std::isfinite(latency_rtt_avg)) {
                    fmt::print("\"avg\": null, "sv);
                } else {
                    fmt::print("\"avg\": {:.9e}, "sv, latency_rtt_avg);
                }
                if (!std::isfinite(latency_rtt_max)) {
                    fmt::print("\"max\": null"sv);
                } else {
                    fmt::print("\"max\": {:.9e}"sv, latency_rtt_max);
                }
                fmt::print("}}\n    }}\n}}\n// Use OMP_NUM_THREADS to customize thread count,\n// use OMP_PLACES or GOMP_CPU_AFFINITY to customize thread affinity.\n"sv);
            }
        }
    }
    return retval;
}

template <>
inline uint64_t benchmark_float_add<float>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] += 1.0f;
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] += -1.0f;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}
template <>
inline uint64_t benchmark_float_add<double>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] += 1.0;
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] += -1.0;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}

template <>
inline uint64_t benchmark_float_mul<float>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 5.0f;
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 0.2f;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}
template <>
inline uint64_t benchmark_float_mul<double>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 5.0;
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 0.2;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}

template <>
inline uint64_t benchmark_float_fma<float>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fmaf(a[j], 5.0f, a[j]);
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fmaf(a[j], -0.8333333f, a[j]);
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 4 * float_batch_size;
}
template <>
inline uint64_t benchmark_float_fma<double>(uint64_t num_batches) {
    constexpr size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fma(a[j], 5.0, a[j]);
        }
#pragma GCC unroll float_batch_size
        for (size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fma(a[j], -0.8333333333333334, a[j]);
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 4 * float_batch_size;
}

static inline void benchmark_mem_write(const BenchStorage &local_storage) {
    void *mem_write_buf = local_storage.mem_write_buf;
    size_t mem_write_size = local_storage.mem_write_size;
    size_t num_mem_writes = local_storage.num_mem_writes;
    for (size_t i = 0; i < num_mem_writes; i++) {
        std::memset(mem_write_buf, (uint8_t) i, mem_write_size);
        benchmark::DoNotOptimize(mem_write_buf);
    }
}

static inline PerfDuration benchmark_latency_host(const BenchStorage &host_storage) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    size_t num_latency_measures = host_storage.num_latency_measures;

    latency_flag.store(true, std::memory_order_release);
    while (latency_flag.load(std::memory_order_acquire)) {
        ia32_pause();
    }
    PerfTimer start = PerfTimer::now();
    latency_flag.store(true, std::memory_order_release);
    for (size_t i = 1; i < num_latency_measures; i++) {
        bool false_value = false;
        while (!latency_flag.compare_exchange_weak(false_value, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
            ia32_pause();
            false_value = false;
        }
    }
    while (latency_flag.load(std::memory_order_acquire)) {
        ia32_pause();
    }
    PerfTimer finish = PerfTimer::now();
    return finish - start;
}

static inline void benchmark_latency_guest(const BenchStorage &host_storage) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    size_t num_latency_measures = host_storage.num_latency_measures;

    for (size_t i = 0; i <= num_latency_measures; i++) {
        bool true_value = true;
        while (!latency_flag.compare_exchange_weak(true_value, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
            ia32_pause();
            true_value = true;
        }
    }
}

static inline void print_affinity(std::span<const std::unique_ptr<BenchStorage>> storage) {
    bool need_comma = false;
    for (size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            fmt::print(
                "        \"{0}\": {{\"omp_thread_num\": {0}, \"omp_affinity\": {1}, \"kernel_affinity\": {2}, \"numa_node\": {3}}}"sv,
                i, storage[i]->omp_affinity, storage[i]->kernel_affinity, storage[i]->numa_node
            );
        }
    }
}

static inline void print_flops(std::span<const std::unique_ptr<BenchStorage>> storage, const PerfDuration BenchStorage::*duration, const uint64_t BenchStorage::*num_float_ops, double &flops_sum) {
    flops_sum = 0;
    bool need_comma = false;
    for (size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            uint64_t num_float_ops_i = storage[i].get()->*num_float_ops;
            const PerfDuration &duration_i = storage[i].get()->*duration;
            fmt::print(
                "        \"{}\": {{\"ops_performed\": {}, \"elapsed\": {}, "sv,
                i, num_float_ops_i, duration_i
            );
            if (num_float_ops_i == 0) {
                flops_sum = std::numeric_limits<double>::quiet_NaN();
                fmt::print("\"flops\": null}}"sv);
            } else {
                double flops = num_float_ops_i / duration_i.seconds().count();
                flops_sum += flops;
                fmt::print("\"flops\": {:.16e}}}"sv, flops);
            }
        }
    }
}

static inline void print_mem_write(std::span<const std::unique_ptr<BenchStorage>> storage, double &throughput_sum) {
    throughput_sum = 0;
    bool need_comma = false;
    for (size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            fmt::print("        \"{}\": {{"sv, i);
            size_t bytes_written;
            if (__builtin_mul_overflow(storage[i]->mem_write_size, storage[i]->num_mem_writes, &bytes_written)) {
                fmt::print("\"bytes_written\": null, "sv);
            } else {
                fmt::print("\"bytes_written\": {}, "sv, bytes_written);
            }
            const PerfDuration &duration = storage[i]->mem_write_duration;
            fmt::print("\"elapsed\": {}, "sv, duration);
            if (storage[i]->mem_write_size == 0 && storage[i]->num_mem_writes == 0) {
                throughput_sum = std::numeric_limits<double>::quiet_NaN();
                fmt::print("\"throughput\": null}}"sv);
            } else {
                double throughput = (double) storage[i]->mem_write_size * storage[i]->num_mem_writes / duration.seconds().count();
                throughput_sum += throughput;
                fmt::print("\"throughput\": {:.16e}}}"sv, throughput);
            }
        }
    }
}

static inline void print_latency(std::span<const std::unique_ptr<BenchStorage>> storage, double &latency_rtt_sum, double &latency_rtt_max) {
    latency_rtt_sum = 0;
    latency_rtt_max = 0;
    bool need_comma = false;
    for (size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            fmt::print("        \"{}\": {{\"rtt_to\": {{"sv, i);
            for (size_t j = 0; j < storage[i]->latency_duration.size(); j++) {
                if (j != 0) {
                    fmt::print(", "sv);
                }
                if (storage[i]->num_latency_measures == 0) {
                    latency_rtt_sum = std::numeric_limits<double>::quiet_NaN();
                    latency_rtt_max = std::numeric_limits<double>::quiet_NaN();
                    fmt::print("\"{}\": null"sv, j);
                } else {
                    double rtt = storage[i]->latency_duration[j].seconds().count() / storage[i]->num_latency_measures;
                    latency_rtt_sum += rtt;
                    if (!(latency_rtt_max >= rtt)) {
                        latency_rtt_max = rtt;
                    }
                    fmt::print("\"{}\": {:.9e}"sv, j, rtt);
                }
            }
            fmt::print("}}}}"sv);
        }
    }
}

} // namespace roofbench
