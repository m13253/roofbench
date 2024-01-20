# roofbench

Benchmark utility for CPU FLOPS, core latency, and memory bandwidth.

## Building

Dependencies:
* Linux
* GCC 11.0 or Clang 11.0 or newer
* LLVM OpenMP Runtime Library (if using Clang)
* Meson build system
* numactl or libnuma-dev

```bash
export CC=gcc CXX=g++
meson setup builddir -D simd_batch_size_f32=232 -D simd_batch_size_f64=116
ninja -C builddir
```
```bash
export CC=clang CXX=clang++
meson setup builddir -D simd_batch_size_f32=240 -D simd_batch_size_f64=120
ninja -C builddir
```

The build system uses `-march=native` by default, so the binary will be optimized for your specific machine.

(Turning on 512-bit SIMD may be fasster on Intel CPUs, if there is only one application running.)
```bash
export CC=gcc CXX=g++
meson setup builddir -D cpp_args=-mprefer-vector-width=512 -D simd_batch_size_f32=464 -D simd_batch_size_f64=232 --wipe
ninja -C builddir
```
```bash
export CC=clang CXX=clang++
meson setup builddir -D cpp_args=-mprefer-vector-width=512 -D simd_batch_size_f32=480 -D simd_batch_size_f64=240 --wipe
ninja -C builddir
```

## Optimal SIMD batch size

On x86, the optimal value is: (total SIMD register count − occupied count) × (SIMD lane width) ÷ sizeof (float).

| Compiler | 256-bit SIMD | 512-bit SIMD |
|:--------:|:------------:|:------------:|
|   GCC    |   232, 116   |   240, 120   |
|  Clang   |   464, 232   |   480, 240   |

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
