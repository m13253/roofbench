#include "bench.hpp"
#include "denorm.hpp"
#include "options.hpp"
#include "pause.hpp"
#include "perf_timer.hpp"
#include <atomic>
#include <benchmark/benchmark.h>
#include <cmath>
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
#include <string_view>
#include <vector>

using namespace std::literals;

namespace roofbench {

struct BenchStorage final {
    int omp_affinity;
    int kernel_affinity;
    int numa_node;

    PerfTimer float_add_finish;
    PerfTimer float_mul_finish;
    PerfTimer float_fma_finish;
    PerfTimer mem_write_finish;
    std::vector<PerfDuration> latency_duration;

    uint64_t num_float_ops_simd;
    size_t mem_write_size;
    void *mem_write_buf;
    void *latency_flag_buf;
    std::atomic_bool *latency_flag;

    std::binary_semaphore latency_host_sem{0};
    std::binary_semaphore latency_guest_sem{0};

    explicit BenchStorage(const AppOptions &options, int num_threads) {
        omp_affinity = omp_get_place_num();
        kernel_affinity = sched_getcpu();
        numa_node = numa_node_of_cpu(kernel_affinity);

        latency_duration = std::vector<PerfDuration>(num_threads);

        num_float_ops_simd = options.num_float_ops == 0 ? 0 : (options.num_float_ops - 1) / (AppOptions::simd_batch_size * 2) + 1;
        size_t pagesize = numa_pagesize();
        mem_write_size = options.mem_write_size == 0 ? 0 : ((options.mem_write_size - 1) / pagesize + 1) * pagesize;
        mem_write_buf = numa_alloc_local(mem_write_size);
        if (!mem_write_buf) {
            throw std::bad_alloc();
        }
        latency_flag_buf = numa_alloc_local(sizeof *latency_flag);
        if (!latency_flag_buf) {
            throw std::bad_alloc();
        }
        std::memset(mem_write_buf, 0xcc, mem_write_size);
        std::memset(latency_flag_buf, 0xcc, sizeof *latency_flag);
        benchmark::DoNotOptimize(mem_write_buf);
        benchmark::DoNotOptimize(latency_flag_buf);
        latency_flag = new (latency_flag_buf) std::atomic_bool(false);
    }
    BenchStorage(const BenchStorage &) = delete;
    BenchStorage &operator=(const BenchStorage &) = delete;
    ~BenchStorage() noexcept {
        numa_free(mem_write_buf, mem_write_size);
        latency_flag->~atomic();
        numa_free(latency_flag_buf, sizeof *latency_flag);
    }
};

static inline void benchmark_float_add(BenchStorage &local_storage);
static inline void benchmark_float_mul(BenchStorage &local_storage);
static inline void benchmark_float_fma(BenchStorage &local_storage);
static inline void benchmark_mem_write(const AppOptions &options, BenchStorage &local_storage);
static inline void benchmark_latency_host(const AppOptions &options, BenchStorage &host_storage, int guest_thread_num);
static inline void benchmark_latency_guest(const AppOptions &options, BenchStorage &host_storage);

int benchmark(const AppOptions &options) {
    int retval = 0;
    std::vector<std::unique_ptr<BenchStorage>> storage;
    double float_add_flops_sum = 0;
    double float_mul_flops_sum = 0;
    double float_fma_flops_sum = 0;
    double mem_write_throughput_sum = 0;
    double latency_rtt_sum = 0;
    double latency_rtt_max = 0;
    std::atomic_int float_add_spinner{0};
    std::atomic_int float_mul_spinner{0};
    std::atomic_int float_fma_spinner{0};
    std::atomic_int mem_write_spinner{0};
    PerfTimer float_add_start;
    PerfTimer float_mul_start;
    PerfTimer float_fma_start;
    PerfTimer mem_write_start;

#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
#pragma omp master
        {
            switch (omp_get_proc_bind()) {
            case omp_proc_bind_false:
            case omp_proc_bind_master:
                fmt::println(stderr, "Error: environment variable OMP_PROC_BIND should be true, close, or spread."sv);
                retval = 1;
                break;
            default:
                fmt::print("{{\n    \"affinity\": {{\n"sv);
                storage = std::vector<std::unique_ptr<BenchStorage>>(num_threads);
                float_add_spinner.store(num_threads, std::memory_order_relaxed);
                float_mul_spinner.store(num_threads, std::memory_order_relaxed);
                float_fma_spinner.store(num_threads, std::memory_order_relaxed);
                mem_write_spinner.store(num_threads, std::memory_order_relaxed);
                numa_set_bind_policy(true);
            }
        }
#pragma omp barrier
        if (retval == 0) {
            enable_denorm_ftz();
            BenchStorage &local_storage = *(storage[thread_num] = std::make_unique<BenchStorage>(options, num_threads)).get();
#pragma omp barrier
#pragma omp master
            {
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
                fmt::print("\n    }},\n    \"float_add\": {{\n"sv);
            }
            if (float_add_spinner.fetch_sub(1, std::memory_order_release) == 1) {
                float_add_start = PerfTimer::now();
            } else {
                while (float_add_spinner.load(std::memory_order_acquire) != 0) {
                    ia32_pause();
                }
            }
            benchmark_float_add(local_storage);
#pragma omp barrier
#pragma omp master
            {
                bool need_comma = false;
                for (size_t i = 0; i < storage.size(); i++) {
                    if (need_comma) {
                        fmt::print(",\n"sv);
                    }
                    need_comma = true;
                    if (!storage[i]) {
                        fmt::print("        \"{}\": null"sv, i);
                    } else {
                        uint64_t num_float_ops = storage[i]->num_float_ops_simd * (AppOptions::simd_batch_size * 2);
                        PerfDuration duration = storage[i]->float_add_finish - float_add_start;
                        fmt::print(
                            "        \"{}\": {{\"ops_performed\": {}, \"elapsed\": {}, "sv,
                            i, num_float_ops, duration
                        );
                        if (num_float_ops == 0) {
                            float_add_flops_sum = std::numeric_limits<double>::quiet_NaN();
                            fmt::print("\"flops\": null}}"sv);
                        } else {
                            double flops = num_float_ops / duration.seconds().count();
                            if (!std::isnan(float_add_flops_sum)) {
                                float_add_flops_sum += flops;
                            }
                            fmt::print("\"flops\": {:.16e}}}"sv, flops);
                        }
                    }
                }
                fmt::print("\n    }},\n    \"float_mul\": {{\n"sv);
            }
            if (float_mul_spinner.fetch_sub(1, std::memory_order_release) == 1) {
                float_mul_start = PerfTimer::now();
            } else {
                while (float_mul_spinner.load(std::memory_order_acquire) != 0) {
                    ia32_pause();
                }
            }
            benchmark_float_mul(local_storage);
#pragma omp barrier
#pragma omp master
            {
                bool need_comma = false;
                for (size_t i = 0; i < storage.size(); i++) {
                    if (need_comma) {
                        fmt::print(",\n"sv);
                    }
                    need_comma = true;
                    if (!storage[i]) {
                        fmt::print("        \"{}\": null"sv, i);
                    } else {
                        uint64_t num_float_ops = storage[i]->num_float_ops_simd * (AppOptions::simd_batch_size * 2);
                        PerfDuration duration = storage[i]->float_mul_finish - float_mul_start;
                        fmt::print(
                            "        \"{}\": {{\"ops_performed\": {}, \"elapsed\": {}, "sv,
                            i, num_float_ops, duration
                        );
                        if (num_float_ops == 0) {
                            float_mul_flops_sum = std::numeric_limits<double>::quiet_NaN();
                            fmt::print("\"flops\": null}}"sv);
                        } else {
                            double flops = num_float_ops / duration.seconds().count();
                            if (!std::isnan(float_mul_flops_sum)) {
                                float_mul_flops_sum += flops;
                            }
                            fmt::print("\"flops\": {:.16e}}}"sv, flops);
                        }
                    }
                }
                fmt::print("\n    }},\n    \"float_fma\": {{\n"sv);
            }
            if (float_fma_spinner.fetch_sub(1, std::memory_order_release) == 1) {
                float_fma_start = PerfTimer::now();
            } else {
                while (float_fma_spinner.load(std::memory_order_acquire) != 0) {
                    ia32_pause();
                }
            }
            benchmark_float_fma(local_storage);
#pragma omp barrier
#pragma omp master
            {
                bool need_comma = false;
                for (size_t i = 0; i < storage.size(); i++) {
                    if (need_comma) {
                        fmt::print(",\n"sv);
                    }
                    need_comma = true;
                    if (!storage[i]) {
                        fmt::print("        \"{}\": null"sv, i);
                    } else {
                        uint64_t num_float_ops = storage[i]->num_float_ops_simd * (AppOptions::simd_batch_size * 4);
                        PerfDuration duration = storage[i]->float_fma_finish - float_fma_start;
                        fmt::print(
                            "        \"{}\": {{\"ops_performed\": {}, \"elapsed\": {}, "sv,
                            i, num_float_ops, duration
                        );
                        if (num_float_ops == 0) {
                            float_fma_flops_sum = std::numeric_limits<double>::quiet_NaN();
                            fmt::print("\"flops\": null}}"sv);
                        } else {
                            double flops = num_float_ops / duration.seconds().count();
                            if (!std::isnan(float_fma_flops_sum)) {
                                float_fma_flops_sum += flops;
                            }
                            fmt::print("\"flops\": {:.16e}}}"sv, flops);
                        }
                    }
                }
                fmt::print("\n    }},\n    \"mem_write\": {{\n"sv);
            }
            if (mem_write_spinner.fetch_sub(1, std::memory_order_release) == 1) {
                mem_write_start = PerfTimer::now();
            } else {
                while (mem_write_spinner.load(std::memory_order_acquire) != 0) {
                    ia32_pause();
                }
            }
            benchmark_mem_write(options, local_storage);
#pragma omp barrier
#pragma omp master
            {
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
                        if (__builtin_mul_overflow(storage[i]->mem_write_size, options.num_mem_writes, &bytes_written)) {
                            fmt::print("\"bytes_written\": null, "sv);
                        } else {
                            fmt::print("\"bytes_written\": {}, "sv, bytes_written);
                        }
                        PerfDuration duration = storage[i]->mem_write_finish - mem_write_start;
                        fmt::print("\"elapsed\": {}, "sv, duration);
                        if (storage[i]->mem_write_size == 0 && options.num_mem_writes == 0) {
                            mem_write_throughput_sum = std::numeric_limits<double>::quiet_NaN();
                            fmt::print("\"throughput\": null}}"sv);
                        } else {
                            double throughput = (double) storage[i]->mem_write_size * options.num_mem_writes / duration.seconds().count();
                            if (!std::isnan(mem_write_throughput_sum)) {
                                mem_write_throughput_sum += throughput;
                            }
                            fmt::print("\"throughput\": {:.16e}}}"sv, throughput);
                        }
                    }
                }
                fmt::print("\n    }},\n    \"inter_thread_latency\": {{\n"sv);
            }
            if (thread_num != 0) {
                for (int i = 0; i < thread_num; i++) {
                    local_storage.latency_guest_sem.acquire();
                    benchmark_latency_guest(options, *storage[i]);
                }
                local_storage.latency_host_sem.acquire();
            }
            for (int i = 0; i < thread_num; i++) {
                storage[i]->latency_guest_sem.release();
                benchmark_latency_host(options, local_storage, i);
            }
            for (int i = thread_num + 1; i < num_threads; i++) {
                storage[i]->latency_guest_sem.release();
                benchmark_latency_host(options, local_storage, i);
            }
            if (thread_num + 1 == num_threads) {
                storage[0]->latency_host_sem.release();
            } else {
                storage[thread_num + 1]->latency_host_sem.release();
                for (int i = thread_num + 1; i < num_threads; i++) {
                    local_storage.latency_guest_sem.acquire();
                    benchmark_latency_guest(options, *storage[i]);
                }
            }
#pragma omp master
            {
                local_storage.latency_host_sem.acquire();
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
                            if (options.num_latency_measures == 0) {
                                latency_rtt_sum = std::numeric_limits<double>::quiet_NaN();
                                latency_rtt_max = std::numeric_limits<double>::quiet_NaN();
                                fmt::print("\"{}\": null"sv, j);
                            } else {
                                double rtt = storage[i]->latency_duration[j].seconds().count() / options.num_latency_measures;
                                if (!std::isnan(latency_rtt_sum)) {
                                    latency_rtt_sum += rtt;
                                }
                                if (!std::isnan(latency_rtt_max) && rtt > latency_rtt_max) {
                                    latency_rtt_max = rtt;
                                }
                                fmt::print("\"{}\": {:.9e}"sv, j, rtt);
                            }
                        }
                        fmt::print("}}}}"sv);
                    }
                }
            }
#pragma omp barrier
            storage[thread_num].reset();
#pragma omp master
            {
                fmt::print("\n    }},\n    \"summary\": {{\n        \"flops\": {{"sv);
                if (std::isnan(float_add_flops_sum)) {
                    fmt::print("\"add\": null, "sv);
                } else {
                    fmt::print("\"add\": {:.16e}, "sv, float_add_flops_sum);
                }
                if (std::isnan(float_mul_flops_sum)) {
                    fmt::print("\"mul\": null, "sv);
                } else {
                    fmt::print("\"mul\": {:.16e}, "sv, float_mul_flops_sum);
                }
                if (std::isnan(float_fma_flops_sum)) {
                    fmt::print("\"fma\": null"sv);
                } else {
                    fmt::print("\"fma\": {:.16e}"sv, float_fma_flops_sum);
                }
                if (std::isnan(mem_write_throughput_sum)) {
                    fmt::print("}},\n        \"mem_write_throughput\": null,\n"sv);
                } else {
                    fmt::print("}},\n        \"mem_write_throughput\": {:.16e},\n"sv, mem_write_throughput_sum);
                }
                fmt::print("        \"inter_thread_latency_rtt\": {{");
                if (std::isnan(latency_rtt_sum)) {
                    fmt::print("\"avg\": null, "sv);
                } else {
                    fmt::print("\"avg\": {:.9e}, "sv, latency_rtt_sum / (num_threads * num_threads));
                }
                if (std::isnan(latency_rtt_max)) {
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

static inline void benchmark_float_add(BenchStorage &local_storage) {
    constexpr size_t simd_batch_size = AppOptions::simd_batch_size;
    uint64_t num_float_ops_simd = local_storage.num_float_ops_simd;
    alignas(4096) float a[simd_batch_size];
    for (size_t j = 0; j < simd_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_float_ops_simd; i++) {
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] += 1.0f;
        }
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] += -1.0f;
        }
    }
    benchmark::DoNotOptimize(a);
    local_storage.float_add_finish = PerfTimer::now();
}

static inline void benchmark_float_mul(BenchStorage &local_storage) {
    constexpr size_t simd_batch_size = AppOptions::simd_batch_size;
    uint64_t num_float_ops_simd = local_storage.num_float_ops_simd;
    alignas(4096) float a[simd_batch_size];
    for (size_t j = 0; j < simd_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_float_ops_simd; i++) {
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] *= 5.0f;
        }
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] *= 0.2f;
        }
    }
    benchmark::DoNotOptimize(a);
    local_storage.float_mul_finish = PerfTimer::now();
}

static inline void benchmark_float_fma(BenchStorage &local_storage) {
    constexpr size_t simd_batch_size = AppOptions::simd_batch_size;
    uint64_t num_float_ops_simd = local_storage.num_float_ops_simd;
    alignas(4096) float a[simd_batch_size];
    for (size_t j = 0; j < simd_batch_size; j++) {
        a[j] = 1.0f;
    }
    benchmark::DoNotOptimize(a);

    for (uint64_t i = 0; i < num_float_ops_simd; i++) {
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] += a[j] * 5.0f;
        }
        for (size_t j = 0; j < simd_batch_size; j++) {
            a[j] += a[j] * -0.8333333f;
        }
    }
    benchmark::DoNotOptimize(a);
    local_storage.float_fma_finish = PerfTimer::now();
}

static inline void benchmark_mem_write(const AppOptions &options, BenchStorage &local_storage) {
    void *mem_write_buf = local_storage.mem_write_buf;
    size_t mem_write_size = local_storage.mem_write_size;
    size_t num_mem_writes = options.num_mem_writes;
    for (size_t i = 0; i < num_mem_writes; i++) {
        std::memset(mem_write_buf, (uint8_t) i, mem_write_size);
        benchmark::DoNotOptimize(mem_write_buf);
    }
    local_storage.mem_write_finish = PerfTimer::now();
}

static inline void benchmark_latency_host(const AppOptions &options, BenchStorage &host_storage, int guest_thread_num) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    size_t num_latency_measures = options.num_latency_measures;

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
    host_storage.latency_duration[guest_thread_num] = finish - start;
}

static inline void benchmark_latency_guest(const AppOptions &options, BenchStorage &host_storage) {
    std::atomic_bool &latency_flag = *host_storage.latency_flag;
    size_t num_latency_measures = options.num_latency_measures;

    for (size_t i = 0; i <= num_latency_measures; i++) {
        bool true_value = true;
        while (!latency_flag.compare_exchange_weak(true_value, false, std::memory_order_acq_rel, std::memory_order_acquire)) {
            ia32_pause();
            true_value = true;
        }
    }
}

} // namespace roofbench
