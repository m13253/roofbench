project(
    'roofbench',
    'cpp',
    version: '0.1',
    license: 'MIT',
    default_options: [
        'b_lto=true',
        'buildtype=debugoptimized',
        'cpp_std=c++20',
        'default_library=static',
        'optimization=3',
        'warning_level=3',
    ],
)

add_project_arguments('-march=native', language: 'cpp')

benchmark = subproject(
    'google-benchmark',
    default_options: ['cpp_std=c++20'],
).get_variable('google_benchmark_dep')
cli11 = subproject(
    'cli11',
    default_options: ['cpp_std=c++20'],
).get_variable('CLI11_dep')
fmt = subproject(
    'fmt',
    default_options: ['cpp_std=c++20'],
).get_variable('fmt_dep')
numa = dependency('numa')
openmp = dependency('openmp', language: 'cpp')
threads = dependency('threads')

sources = [
    'src/bench.cpp',
    'src/main.cpp',
]

executable(
    'roofbench',
    sources,
    cpp_args: [
        '-DROOFBENCH_SIMD_BATCH_SIZE=@0@'.format(get_option('simd_batch_size')),
    ],
    install: true,
    dependencies: [
        benchmark,
        cli11,
        fmt,
        numa,
        openmp,
        threads,
    ],
)
