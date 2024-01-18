# roofbench

Benchmark utility for CPU FLOPS, core latency, and memory bandwidth.

## Building

Dependencies:
* Linux
* GCC 11.0 or Clang 11.0 or newer
* LLVM OpenMP Runtime Library (if using Clang)
* numactl or libnuma-dev

```bash
meson setup builddir -Dsimd_batch_size=512
ninja -C builddir
```

## Running

```bash
OMP_PLACES=threads OMP_PROC_BIND=true ./builddir/roofbench
```

The output is in JSON format.

## Included benchmarks

1. Affinity: shows thread affinity
2. Float Add: floating-point add operations
3. Float Mul: floating-point multiply operation
4. Mem Write: memset into corresponding NUMA local memory
5. Inter-thread Latency: round-trip time between each pair of host thread and guest thread, through shared memory communication on host thread’s NUMA node

## Units of measurement

* Time duration: seconds
* FLOPS: operations per second
* Throughput: bytes per second
* Latency: seconds

## License

The program is free and open-source software, licensed under the MIT license.

Refer to the [LICENSE](LICENSE) file for more information.
