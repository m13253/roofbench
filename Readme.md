# roofbench

Benchmark utility for CPU FLOPS, core latency, and memory bandwidth.

## Building

Dependencies:
* Linux
* Clang 11.0 or newer

  (depending on software version, GCC may not properly generate SIMD instructions)
* LLVM OpenMP Runtime Library (if using Clang)
* Meson build system
* numactl or libnuma-dev

```bash
export CC=clang CXX=clang++
meson setup builddir -D simd_batch_size=240
ninja -C builddir
```

(Turning on 512-bit SIMD may be fasster on Intel CPUs.)
```bash
meson setup builddir -D cpp_args=-mprefer-vector-width=512 -D simd_batch_size=464
```

## Running

```bash
OMP_PLACES=threads OMP_PROC_BIND=true ./builddir/roofbench
```

The output is in JSON format.

## Included benchmarks

1. Affinity: shows thread affinity
2. Float Add: floating-point add operations
3. Float Mul: floating-point multiply operations
4. Float FMA: fused floating-point multiply then add operations
5. Mem Write: memset into corresponding NUMA local memory
6. Inter-thread Latency: round-trip time between each pair of host thread and guest thread, through shared memory communication on host thread’s NUMA node

## Units of measurement

* Time duration: seconds
* FLOPS: operations per second
* Throughput: bytes per second
* Latency: seconds

## License

The program is free and open-source software, licensed under the MIT license.

Refer to the [LICENSE](LICENSE) file for more information.
