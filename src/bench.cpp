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
#include <cstdlib>
#include <cstring>
#include <fmt/core.h>
#include <limits>
#include <memory>
#include <new>
#ifdef __linux__
#include <numa.h>
#endif
#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <semaphore>
#include <span>
#include <string_view>
#if defined(__linux__) && (__GLIBC__ < 2 || __GLIBC__ == 2 && __GLIBC_MINOR__ < 29)
#include <sys/syscall.h> // glibc < 2.29 didn't have getcpu() in sched.h
#include <unistd.h>
#endif
#include <system_error>
#include <vector>

using namespace std::literals;

namespace roofbench {

struct BenchStorage final {
    int omp_affinity;
    int kernel_affinity;
    int numa_node;

    std::uint64_t num_float_batches_f32;
    std::uint64_t num_float_batches_f64;
    std::uint64_t num_float_add_ops_f32;
    std::uint64_t num_float_mul_ops_f32;
    std::uint64_t num_float_fma_ops_f32;
    std::uint64_t num_float_add_ops_f64;
    std::uint64_t num_float_mul_ops_f64;
    std::uint64_t num_float_fma_ops_f64;
    PerfDuration float_add_duration_f32;
    PerfDuration float_mul_duration_f32;
    PerfDuration float_fma_duration_f32;
    PerfDuration float_add_duration_f64;
    PerfDuration float_mul_duration_f64;
    PerfDuration float_fma_duration_f64;

    PerfDuration mem_read_duration;
    std::vector<PerfDuration> latency_duration;

    std::size_t pagesize;
    std::size_t mem_read_size;
    std::size_t num_mem_reads;
    std::size_t num_latency_measures;

    void *mem_read_buf;
    void *latency_flag_buf;

    std::atomic_bool *latency_flag;
    std::binary_semaphore latency_host_sem{0};
    std::binary_semaphore latency_guest_sem{0};

    explicit BenchStorage(const AppOptions &options, int num_threads) {
        omp_affinity = omp_get_place_num();
#ifdef __linux__
#if __GLIBC__ < 2 || __GLIBC__ == 2 && __GLIBC_MINOR__ < 29
        if (syscall(SYS_getcpu, (unsigned *) &kernel_affinity, (unsigned *) &numa_node, nullptr) != 0) { // glibc < 2.29 didn't have getcpu() in sched.h
            throw std::system_error(errno, std::generic_category());
        }
#else
        if (getcpu((unsigned *) &kernel_affinity, (unsigned *) &numa_node) != 0) {
            throw std::system_error(errno, std::generic_category());
        }
#endif
#else
        kernel_affinity = -1;
        numa_node = -1;
#endif

        num_float_batches_f32 = div_ceil<std::uint64_t>(options.num_float_ops_f32, AppOptions::float_batch_size<float> * (std::uint64_t) 2);
        num_float_batches_f64 = div_ceil<std::uint64_t>(options.num_float_ops_f64, AppOptions::float_batch_size<double> * (std::uint64_t) 2);
        latency_duration = std::vector<PerfDuration>(num_threads);

#ifdef __linux__
        pagesize = numa_pagesize();
#else
        pagesize = 4096;
#endif
        mem_read_size = div_ceil(options.mem_read_size, pagesize) * pagesize;
        num_mem_reads = options.num_mem_reads;
        num_latency_measures = options.num_latency_measures;

#ifdef __linux__
        latency_flag_buf = numa_alloc_local(pagesize + mem_read_size);
#else
        latency_flag_buf = new std::byte[pagesize + mem_read_size];
#endif
        if (!latency_flag_buf) {
            throw std::bad_alloc();
        }
        mem_read_buf = &((std::byte *) latency_flag_buf)[pagesize];
        std::memset(latency_flag_buf, 0xcc, pagesize + mem_read_size);
        benchmark::DoNotOptimize(latency_flag_buf);

        latency_flag = new (latency_flag_buf) std::atomic_bool(false);
    }

    BenchStorage(const BenchStorage &) = delete;
    BenchStorage &operator=(const BenchStorage &) = delete;

    ~BenchStorage() noexcept {
        latency_flag->~atomic();
#ifdef __linux__
        numa_free(latency_flag_buf, pagesize + mem_read_size);
#else
        delete[] (std::byte *) latency_flag_buf;
#endif
    }
};

template <std::floating_point T>
static inline std::uint64_t benchmark_float_add(std::uint64_t num_batches);
template <std::floating_point T>
static inline std::uint64_t benchmark_float_mul(std::uint64_t num_batches);
template <std::floating_point T>
static inline std::uint64_t benchmark_float_fma(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_add<float>(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_mul<float>(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_fma<float>(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_add<double>(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_mul<double>(std::uint64_t num_batches);
template <>
inline std::uint64_t benchmark_float_fma<double>(std::uint64_t num_batches);

static inline std::size_t benchmark_mem_read(const BenchStorage &local_storage);
static inline PerfDuration benchmark_latency_host(const BenchStorage &host_storage);
static inline void benchmark_latency_guest(const BenchStorage &host_storage);

static inline void print_affinity(std::span<const std::unique_ptr<BenchStorage>> storage);
static inline void print_flops(std::span<const std::unique_ptr<BenchStorage>> storage, const PerfDuration BenchStorage::*duration, const std::uint64_t BenchStorage::*num_float_ops, double &flops_sum);
static inline void print_mem_read(std::span<const std::unique_ptr<BenchStorage>> storage, double &throughput_sum);
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
    double mem_read_throughput_sum;
    double latency_rtt_sum;
    double latency_rtt_max;
    std::array<SpinBarrier, 14> spin_barriers;

#pragma omp parallel
    {
#pragma omp master
        switch (omp_get_proc_bind()) {
        default: {
            const char *gomp_cpu_affinity = std::getenv("GOMP_CPU_AFFINITY");
            if (gomp_cpu_affinity == nullptr || gomp_cpu_affinity[0] == '\0') {
                fmt::print(stderr, "Error: environment variable OMP_PROC_BIND should be true, close, or spread.\n"sv);
                retval = 1;
            }
        }
        case omp_proc_bind_true:
        case omp_proc_bind_close:
        case omp_proc_bind_spread:;
        }
#pragma omp barrier
        if (retval == 0) {
            enable_denorm_ftz();
#ifdef __linux__
            numa_set_localalloc();
            numa_set_strict(true);
#endif
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
#pragma omp master
            {
#ifndef __linux__
                fmt::print(stderr, "Warning: unsupported operating system. Thread affinity and NUMA-aware allocator is unavailable.\n"sv);
#endif
                fmt::print(stderr, "Info: use OMP_NUM_THREADS to customize thread count, use OMP_PLACES or GOMP_CPU_AFFINITY to customize thread affinity.\n"sv);
                std::fflush(stderr);
                fmt::print("{{\n    \"affinity\": {{\n"sv);
                std::fflush(stdout);
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
                std::fflush(stdout);
            }
            spin_barriers[0].wait();
            PerfTimer start = PerfTimer::now();
            spin_barriers[1].wait();
            std::uint64_t num_float_ops = benchmark_float_add<float>(local_storage.num_float_batches_f32);
            PerfTimer finish = PerfTimer::now();
            local_storage.num_float_add_ops_f32 = num_float_ops;
            local_storage.float_add_duration_f32 = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_flops(storage, &BenchStorage::float_add_duration_f32, &BenchStorage::num_float_add_ops_f32, float_add_flops_sum_f32);
                fmt::print("\n    }},\n    \"f32_mul\": {{\n"sv);
                std::fflush(stdout);
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
                std::fflush(stdout);
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
                std::fflush(stdout);
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
                std::fflush(stdout);
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
                std::fflush(stdout);
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
                fmt::print("\n    }},\n    \"mem_read\": {{\n"sv);
                std::fflush(stdout);
            }
            spin_barriers[12].wait();
            start = PerfTimer::now();
            spin_barriers[13].wait();
            local_storage.mem_read_size = benchmark_mem_read(local_storage);
            finish = PerfTimer::now();
            local_storage.mem_read_duration = finish - start;
#pragma omp barrier
#pragma omp master
            {
                print_mem_read(storage, mem_read_throughput_sum);
                fmt::print("\n    }},\n    \"inter_thread_latency\": {{\n"sv);
                std::fflush(stdout);
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
                if (!std::isfinite(mem_read_throughput_sum)) {
                    fmt::print("}},\n        \"mem_read_throughput\": null,\n"sv);
                } else {
                    fmt::print("}},\n        \"mem_read_throughput\": {:.16e},\n"sv, mem_read_throughput_sum);
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
                fmt::print("}}\n    }}\n}}\n"sv);
                std::fflush(stdout);
#ifndef __linux__
                fmt::print(stderr, "Warning: unsupported operating system. Thread affinity and NUMA-aware allocator is unavailable.\n"sv);
#endif
                fmt::print(stderr, "Info: use OMP_NUM_THREADS to customize thread count, use OMP_PLACES or GOMP_CPU_AFFINITY to customize thread affinity.\n"sv);
                std::fflush(stderr);
            }
        }
    }
    return retval;
}

template <>
inline std::uint64_t benchmark_float_add<float>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] += 1.0f;
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] += -1.0f;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}
template <>
inline std::uint64_t benchmark_float_add<double>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] += 1.0;
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] += -1.0;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}

template <>
inline std::uint64_t benchmark_float_mul<float>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 5.0f;
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 0.2f;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}
template <>
inline std::uint64_t benchmark_float_mul<double>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 5.0;
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] *= 0.2;
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 2 * float_batch_size;
}

template <>
inline std::uint64_t benchmark_float_fma<float>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<float>;
    alignas(4096) float a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fmaf(a[j], 5.0f, a[j]);
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fmaf(a[j], -0.8333333f, a[j]);
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 4 * float_batch_size;
}
template <>
inline std::uint64_t benchmark_float_fma<double>(std::uint64_t num_batches) {
    constexpr std::size_t float_batch_size = AppOptions::float_batch_size<double>;
    alignas(4096) double a[float_batch_size];
    for (std::size_t j = 0; j < float_batch_size; j++) {
        a[j] = 1.0;
    }
    benchmark::DoNotOptimize(a);

    for (std::uint64_t i = 0; i < num_batches; i++) {
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fma(a[j], 5.0, a[j]);
        }
#pragma GCC unroll float_batch_size
        for (std::size_t j = 0; j < float_batch_size; j++) {
            a[j] = std::fma(a[j], -0.8333333333333334, a[j]);
        }
    }
    benchmark::DoNotOptimize(a);
    return num_batches * 4 * float_batch_size;
}

static inline std::size_t benchmark_mem_read(const BenchStorage &local_storage) {
    const std::size_t *mem_read_buf = (const std::size_t *) local_storage.mem_read_buf;
    std::size_t mem_read_size = local_storage.mem_read_size;
    std::size_t num_mem_reads = local_storage.num_mem_reads;
    std::size_t sum = 0;
    for (std::size_t i = 0; i < num_mem_reads; i++) {
        for (std::size_t j = 0; j < mem_read_size / sizeof(std::size_t); j++) {
            sum ^= mem_read_buf[j];
        }
        benchmark::DoNotOptimize(sum);
    }
    return mem_read_size / sizeof(std::size_t) * sizeof(std::size_t);
}

static inline PerfDuration benchmark_latency_host(const BenchStorage &host_storage) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    std::size_t num_latency_measures = host_storage.num_latency_measures;

    latency_flag.store(true, std::memory_order_release);
    do {
        ia32_pause();
    } while (latency_flag.load(std::memory_order_acquire));
    PerfTimer start = PerfTimer::now();
    if (num_latency_measures != 0) {
        latency_flag.store(true, std::memory_order_release);
        for (std::size_t i = 1; i < num_latency_measures; i++) {
            bool false_value = false;
            while (!latency_flag.compare_exchange_weak(false_value, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
                ia32_pause();
                false_value = false;
            }
        }
        while (latency_flag.load(std::memory_order_acquire)) {
            ia32_pause();
        }
    }
    PerfTimer finish = PerfTimer::now();
    return finish - start;
}

static inline void benchmark_latency_guest(const BenchStorage &host_storage) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    std::size_t num_latency_measures = host_storage.num_latency_measures;

    for (std::size_t i = 0; i <= num_latency_measures; i++) {
        bool true_value = true;
        while (!latency_flag.compare_exchange_weak(true_value, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
            ia32_pause();
            true_value = true;
        }
    }
}

static inline void print_affinity(std::span<const std::unique_ptr<BenchStorage>> storage) {
    bool need_comma = false;
    for (std::size_t i = 0; i < storage.size(); i++) {
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

static inline void print_flops(std::span<const std::unique_ptr<BenchStorage>> storage, const PerfDuration BenchStorage::*duration, const std::uint64_t BenchStorage::*num_float_ops, double &flops_sum) {
    flops_sum = 0;
    bool need_comma = false;
    for (std::size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            std::uint64_t num_float_ops_i = storage[i].get()->*num_float_ops;
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

static inline void print_mem_read(std::span<const std::unique_ptr<BenchStorage>> storage, double &throughput_sum) {
    throughput_sum = 0;
    bool need_comma = false;
    for (std::size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            fmt::print("        \"{}\": {{"sv, i);
            std::size_t bytes_read;
            if (__builtin_mul_overflow(storage[i]->mem_read_size, storage[i]->num_mem_reads, &bytes_read)) {
                fmt::print("\"bytes_read\": null, "sv);
            } else {
                fmt::print("\"bytes_read\": {}, "sv, bytes_read);
            }
            const PerfDuration &duration = storage[i]->mem_read_duration;
            fmt::print("\"elapsed\": {}, "sv, duration);
            if (storage[i]->mem_read_size == 0 || storage[i]->num_mem_reads == 0) {
                throughput_sum = std::numeric_limits<double>::quiet_NaN();
                fmt::print("\"throughput\": null}}"sv);
            } else {
                double throughput = (double) storage[i]->mem_read_size * storage[i]->num_mem_reads / duration.seconds().count();
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
    for (std::size_t i = 0; i < storage.size(); i++) {
        if (need_comma) {
            fmt::print(",\n"sv);
        }
        need_comma = true;
        if (!storage[i]) {
            fmt::print("        \"{}\": null"sv, i);
        } else {
            fmt::print("        \"{}\": {{\"rtt_to\": {{"sv, i);
            for (std::size_t j = 0; j < storage[i]->latency_duration.size(); j++) {
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
