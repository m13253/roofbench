# roofbench

Benchmark utility for CPU FLOPS, core latency, and memory bandwidth.

## Building

Dependencies:
* Linux
* GCC 11.0 or Clang 11.0 or newer (Clang is preferred)
  * `libomp-dev` LLVM OpenMP Runtime Library (if using Clang)
  * `libc++-dev` LLVM C++ standard library (if using `-stdlib=libc++`)
  * `lld` LLVM linker (if using `-fuse-ld=lld`)
* Meson build system
* numactl or libnuma-dev

```bash
export AR=gcc-ar CC=gcc CXX=g++ RANLIB=gcc-ranlib
meson setup builddir -D simd_batch_size_f32=232 -D simd_batch_size_f64=116
ninja -C builddir
```
```bash
export AR=llvm-ar CC=clang CXX=clang++ RANLIB=llvm-ranlib
export CXXFLAGS=-stdlib=libc++ LDFLAGS='-fuse-ld=lld -stdlib=libc++'  # Optional
meson setup builddir -D simd_batch_size_f32=240 -D simd_batch_size_f64=120
ninja -C builddir
```

The build system uses `-march=native` by default, so the binary will be optimized for your specific machine.

### Intel AVX-512

Turning on 512-bit SIMD can increase peak FLOPS on Intel CPUs. However, in a multitasking environment, the performance of other processes will be reduced.

```bash
export AR=gcc-ar CC=gcc CXX=g++ RANLIB=gcc-ranlib
meson setup builddir -D cpp_args=-mprefer-vector-width=512 -D simd_batch_size_f32=464 -D simd_batch_size_f64=232 --wipe
ninja -C builddir
```
```bash
export AR=llvm-ar CC=clang CXX=clang++ RANLIB=llvm-ranlib
export CXXFLAGS=-stdlib=libc++ LDFLAGS='-fuse-ld=lld -stdlib=libc++'  # Optional
meson setup builddir -D cpp_args="$CXXFLAGS -mprefer-vector-width=512" -D simd_batch_size_f32=480 -D simd_batch_size_f64=240 --wipe
ninja -C builddir
```

### Optimal SIMD batch size

The optimal value is: (total SIMD register count − occupied count) × (SIMD lane width) ÷ sizeof (float).

| Compiler | AArch64 NEON (128-bit) | AVX2 (256-bit) | AVX-512 (512-bit) |
|:--------:|:----------------------:|:--------------:|:-----------------:|
|   GCC    |        120, 60         |    232, 116    |      464, 232     |
|  Clang   |        120, 60         |    240, 120    |      480, 240     |

## Running

```bash
OMP_PLACES=threads OMP_PROC_BIND=true ./builddir/roofbench | tee results.json
./plot_latency.py results.json > latency.svg
```

The output is in JSON format.

### Included benchmarks

1. Affinity: shows thread affinity
2. Float Add: floating-point add operations
3. Float Mul: floating-point multiply operations
4. Float FMA: fused floating-point multiply then add operations
5. Memory Read: reading the corresponding NUMA local memory
6. Inter-thread Latency: round-trip time between each pair of host thread and guest thread, through shared memory communication on host thread’s NUMA node

### Units of measurement

* Time duration: seconds
* FLOPS: operations per second
* Throughput: bytes per second
* Latency: seconds

## License

The program is free and open-source software, licensed under the MIT license.

Refer to the [LICENSE](LICENSE) file for more information.
