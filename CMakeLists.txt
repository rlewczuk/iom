cmake_minimum_required(VERSION 3.21)
option(SYCL_ENABLED "Build with SYCL support" OFF)
set(SYCL_COMPILER "" CACHE FILEPATH
    "SYCL compiler executable (icpx or dpcpp)")

if(SYCL_ENABLED)
    if(SYCL_COMPILER)
        if(NOT EXISTS "${SYCL_COMPILER}")
            message(FATAL_ERROR
                "SYCL compiler does not exist: ${SYCL_COMPILER}")
        endif()
        set(IOM_SYCL_COMPILER "${SYCL_COMPILER}")
    else()
        find_program(IOM_SYCL_COMPILER NAMES icpx dpcpp)
    endif()

    if(NOT IOM_SYCL_COMPILER)
        message(FATAL_ERROR
            "SYCL_ENABLED=ON requires the Intel oneAPI DPC++ compiler "
            "(icpx or dpcpp) in PATH; set SYCL_COMPILER to its executable")
    endif()

    get_filename_component(IOM_SYCL_COMPILER_BIN
        "${IOM_SYCL_COMPILER}" DIRECTORY)
    get_filename_component(IOM_SYCL_ROOT
        "${IOM_SYCL_COMPILER_BIN}/.." ABSOLUTE)
    find_path(IOM_SYCL_INCLUDE_DIR
        NAMES sycl/sycl.hpp
        HINTS "${IOM_SYCL_ROOT}/include")
    find_library(IOM_SYCL_RUNTIME_LIBRARY
        NAMES sycl
        HINTS "${IOM_SYCL_ROOT}/lib")
    if(NOT IOM_SYCL_INCLUDE_DIR OR NOT IOM_SYCL_RUNTIME_LIBRARY)
        message(FATAL_ERROR
            "SYCL_ENABLED=ON requires sycl/sycl.hpp and libsycl; "
            "searched under ${IOM_SYCL_ROOT}")
    endif()

    if(DEFINED CMAKE_CXX_COMPILER)
        get_filename_component(IOM_CONFIGURED_CXX_NAME
            "${CMAKE_CXX_COMPILER}" NAME)
        if(NOT IOM_CONFIGURED_CXX_NAME STREQUAL "icpx"
                AND NOT IOM_CONFIGURED_CXX_NAME STREQUAL "dpcpp")
            message(FATAL_ERROR
                "SYCL_ENABLED=ON requires an Intel oneAPI DPC++ compiler "
                "(icpx or dpcpp); the configured compiler is "
                "${CMAKE_CXX_COMPILER}")
        endif()
    else()
        set(CMAKE_CXX_COMPILER "${IOM_SYCL_COMPILER}" CACHE FILEPATH
            "C++ compiler selected for SYCL-enabled builds")
    endif()
endif()

project(iom
    VERSION 0.0.1
    DESCRIPTION "Inference of Oversized Models"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json REQUIRED)

option(CUDA_ENABLED "Build with CUDA support" OFF)
set(CUDA_PATH "/usr/local/cuda" CACHE PATH "Path to the CUDA Toolkit")
option(ROCM_ENABLED "Build with ROCm support" OFF)
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm SDK")
option(AVX_512_BF16_ENABLED
    "Build explicitly registered CPU sources with isolated AVX-512 BF16 target options"
    OFF)

# The AVX-512 BF16 acceleration path is opt-in and stays out of every portable
# configuration: the option is OFF by default, and its ISA options are applied
# per source rather than to libiom as a whole. Consequently the capability
# probes below run only when the option is ON; a toolchain, target
# architecture, or command line that cannot support the isolated sources fails
# here with a reason instead of producing a library that mixes ISA targets.
if(AVX_512_BF16_ENABLED)
    include(CheckCXXCompilerFlag)
    include(CheckCXXSourceCompiles)

    # The exact option list iom_add_avx512_bf16_source() applies to a
    # registered source. AVX512BF16 builds on the AVX512F vector facility, so
    # both are requested explicitly; nothing else is enabled globally.
    set(IOM_AVX512_BF16_COMPILE_OPTIONS "-mavx512f;-mavx512bf16")

    # 1. Target architecture: AVX-512 exists only on x86-64, and the resulting
    #    objects are linked into the host library, so a non-x86-64 target is a
    #    configuration error rather than a compile failure deep in a source.
    check_cxx_source_compiles("
        #if !defined(__x86_64__) && !defined(_M_X64)
        #error \"AVX-512 BF16 requires an x86-64 target\"
        #endif
        int main() { return 0; }
    " IOM_AVX512_BF16_TARGET_IS_X86_64)
    if(NOT IOM_AVX512_BF16_TARGET_IS_X86_64)
        message(FATAL_ERROR
            "AVX_512_BF16_ENABLED=ON requires an x86-64 target architecture; "
            "the configured compiler targets '${CMAKE_SYSTEM_PROCESSOR}'. "
            "Configure with AVX_512_BF16_ENABLED=OFF for the portable build.")
    endif()

    # 2. Compiler support for the target options themselves.
    check_cxx_compiler_flag("-mavx512f" IOM_AVX512_BF16_HAS_AVX512F_OPTION)
    check_cxx_compiler_flag("-mavx512bf16"
        IOM_AVX512_BF16_HAS_AVX512BF16_OPTION)
    if(NOT IOM_AVX512_BF16_HAS_AVX512F_OPTION
            OR NOT IOM_AVX512_BF16_HAS_AVX512BF16_OPTION)
        message(FATAL_ERROR
            "AVX_512_BF16_ENABLED=ON requires a compiler that accepts the "
            "-mavx512f and -mavx512bf16 target options; "
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not "
            "accept them. Configure with AVX_512_BF16_ENABLED=OFF for the "
            "portable build.")
    endif()

    # 3. The required intrinsics: the BF16 vector packing types, the paired
    #    BF16 dot product into FP32, and the FP32-to-BF16 packing conversion
    #    the kernels depend on, compiled with exactly the per-source option
    #    list above.
    string(REPLACE ";" " " IOM_AVX512_BF16_REQUIRED_FLAGS "${IOM_AVX512_BF16_COMPILE_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS "${IOM_AVX512_BF16_REQUIRED_FLAGS}")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 values = _mm512_set1_ps(1.0f);
            __m512bh packed = _mm512_cvtne2ps_pbh(values, values);
            __m512 dots = _mm512_dpbf16_ps(values, packed, packed);
            return _mm512_cvtsi512_si32(_mm512_castps_si512(dots)) == 0;
        }
    " IOM_AVX512_BF16_HAS_INTRINSICS)
    if(NOT IOM_AVX512_BF16_HAS_INTRINSICS)
        message(FATAL_ERROR
            "AVX_512_BF16_ENABLED=ON requires the AVX512_BF16 intrinsics "
            "_mm512_cvtne2ps_pbh and _mm512_dpbf16_ps to compile with "
            "'${IOM_AVX512_BF16_COMPILE_OPTIONS}'; they do not. Configure with "
            "AVX_512_BF16_ENABLED=OFF for the portable build.")
    endif()

    # 4. Target isolation: the shared baseline command line must not already
    #    target AVX-512, otherwise the per-source options are decorative and
    #    every baseline object would acquire AVX-512 code generation. MSVC
    #    spells the same fact with /arch:AVX512, so its predefined macro is
    #    probed too.
    set(CMAKE_REQUIRED_FLAGS "")
    check_cxx_source_compiles("
        #if defined(__AVX512F__) || defined(__AVX512BF16__) \\
                || defined(_M_AVX512)
        #error \"the baseline command line already enables AVX-512\"
        #endif
        int main() { return 0; }
    " IOM_AVX512_BF16_BASELINE_IS_PORTABLE)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT IOM_AVX512_BF16_BASELINE_IS_PORTABLE)
        message(FATAL_ERROR
            "AVX_512_BF16_ENABLED=ON requires AVX-512 BF16 code generation to "
            "stay behind the per-source target options, but "
            "${CMAKE_CXX_COMPILER} enables AVX-512 for every translation unit "
            "(configured flags: '${CMAKE_CXX_FLAGS}'). Remove the global ISA "
            "options or configure with AVX_512_BF16_ENABLED=OFF.")
    endif()
endif()

set(IOM_SOURCES
        src/device_ops.cpp
        src/device_ops_copy.cpp
        src/device_ops_binary.cpp
        src/device_ops_cache_append.cpp
        src/device_ops_rmsnorm.cpp
        src/device_ops_embedding.cpp
        src/device_ops_neural.cpp
        src/workspace.cpp
        src/tensor.cpp
        src/tensor_view.cpp
        src/token_selection.cpp
        src/cpu/device.cpp
        src/cpu/avx512_bf16_dispatch.cpp
        src/cpu/tensor.cpp
        src/cpu/sdpa.cpp
        src/cpu/queue.cpp
        src/mmap.cpp
        src/safetensors.cpp
        src/tokenizer.cpp
        src/model.cpp
        src/session.cpp
        src/chat_format.cpp
        src/alloc.cpp
        src/inference_metrics.cpp
)

set(IOM_HEADERS
        include/iom/iom.hpp
        include/iom/detail/staged_worker.hpp
        include/iom/oid.hpp
        include/iom/cpu/device.hpp
        include/iom/mmap.hpp
        include/iom/safetensors.hpp
        include/iom/chat_format.hpp
        include/iom/tokenizer.hpp
        include/iom/model.hpp
        include/iom/alloc.hpp
        include/iom/device.hpp
        include/iom/tensor.hpp
        include/iom/token_selection.hpp
        include/iom/session.hpp
        include/iom/inference_metrics.hpp
)


add_library(libiom STATIC
        ${IOM_SOURCES}
        ${IOM_HEADERS}
)

set_target_properties(libiom PROPERTIES
    OUTPUT_NAME iom
)

# The definition marks a build whose isolated AVX-512 BF16 sources exist. It
# is private to libiom: no other target, and no installed interface, may
# depend on the optional path's presence.
if(AVX_512_BF16_ENABLED)
    target_compile_definitions(libiom
        PRIVATE
            IOM_AVX512_BF16_COMPILED
    )
endif()

# Register one existing CPU translation unit as an isolated AVX-512 BF16
# source of libiom. Only the named source receives the AVX512F/AVX512BF16
# target options and the file is added directly to libiom, so no auxiliary
# object or helper library is created and every other libiom source keeps the
# portable baseline command line. The name must be a source that exists in the
# source tree and is not already part of the baseline list; a source that does
# not exist yet cannot be registered ahead of its implementation.
function(iom_add_avx512_bf16_source source)
    if("${source}" STREQUAL "")
        message(FATAL_ERROR
            "iom_add_avx512_bf16_source() requires a source argument")
    endif()
    if(NOT AVX_512_BF16_ENABLED)
        message(FATAL_ERROR
            "iom_add_avx512_bf16_source(${source}) requires "
            "AVX_512_BF16_ENABLED=ON; a portable build must not register an "
            "AVX-512 BF16 source.")
    endif()

    if(IS_ABSOLUTE "${source}")
        set(iom_avx512_bf16_source_path "${source}")
    else()
        set(iom_avx512_bf16_source_path "${PROJECT_SOURCE_DIR}/${source}")
    endif()
    if(NOT EXISTS "${iom_avx512_bf16_source_path}"
            OR IS_DIRECTORY "${iom_avx512_bf16_source_path}")
        message(FATAL_ERROR
            "iom_add_avx512_bf16_source() requires an existing source file; "
            "'${source}' resolves to '${iom_avx512_bf16_source_path}'")
    endif()
    foreach(iom_baseline_source IN LISTS IOM_SOURCES)
        get_filename_component(iom_baseline_source_path
            "${PROJECT_SOURCE_DIR}/${iom_baseline_source}" REALPATH)
        if(iom_avx512_bf16_source_path STREQUAL
                iom_baseline_source_path)
            message(FATAL_ERROR
                "iom_add_avx512_bf16_source() cannot register the baseline "
                "source '${source}'; baseline sources keep the portable "
                "command line")
        endif()
    endforeach()

    target_sources(libiom
        PRIVATE
            "${iom_avx512_bf16_source_path}"
    )
    set_source_files_properties("${iom_avx512_bf16_source_path}" PROPERTIES
        COMPILE_OPTIONS "${IOM_AVX512_BF16_COMPILE_OPTIONS}"
    )
endfunction()

# The isolated AVX-512 BF16 sources, one per accelerated operation. A source is
# registered only once its implementation exists: a portable configuration
# registers nothing, and an enabled configuration compiles exactly these files
# with the target options above.
if(AVX_512_BF16_ENABLED)
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_rmsnorm_reduction.cpp)
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_silu.cpp)
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_binary_mul.cpp)
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_rmsnorm_store.cpp)
    # BF16 RoPE split-half pair arithmetic.
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_rope.cpp)
    iom_add_avx512_bf16_source(src/cpu/avx512_bf16_linear.cpp)
endif()


target_include_directories(libiom
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(libiom
    PUBLIC
        nlohmann_json::nlohmann_json
)

if(CUDA_ENABLED)
    set(CUDAToolkit_ROOT "${CUDA_PATH}")
    find_package(CUDAToolkit REQUIRED MODULE)
    enable_language(CUDA)
    add_library(iom_cuda STATIC
        src/cuda/device.cpp
        src/cuda/device_tensor.cpp
        src/cuda/device_internal.hpp
        src/cuda/copy.cu
        src/cuda/copy.hpp
        src/cuda/sdpa.cpp
        src/cuda/sdpa.hpp
        src/cuda/sdpa_nonmatrix.cu
        src/cuda/sdpa_nonmatrix.hpp
        src/cuda/sdpa_nonmatrix_request.hpp
        src/cuda/linear.cu
        src/cuda/sdpa_matrix.cu
        src/cuda/sdpa_matrix.hpp
        src/cuda/sdpa_matrix_request.hpp
        src/cuda/driver.cpp
        src/cuda/driver.hpp
        src/shared/metadata_slot_pool.hpp
        src/shared/event_ring.hpp
        src/shared/gpu_queue.hpp
        src/shared/gpu_queue_lifecycle.inl
        src/shared/gpu_queue_operations.inl
        src/shared/standard_tiled_copy.hpp
        src/shared/standard_tiled_copy.inl
        src/shared/standard_tiled_embedding.hpp
        src/shared/standard_tiled_embedding.inl
        src/shared/standard_tiled_copy_metadata.inl
        src/shared/standard_tiled_linear.inl
        src/shared/standard_tiled_rope.inl
        src/shared/scalar_binary_codec.hpp
        include/iom/cuda/device.hpp
    )

    # The native BF16 linear specialization (src/cuda/linear.cu) is a direct
    # `wmma` BF16-input/FP32-accumulate route, and its device statements exist
    # only in images of an architecture at or above the documented BF16 WMMA
    # floor. The toolkit default is a single pre-floor PTX image, so the archs
    # are pinned here: `80` is the floor, `90` keeps the accelerated image for
    # Hopper, `120` covers GeForce Blackwell, and `75` keeps the images the
    # previous default produced for the scalar path. Covered families are
    # therefore Turing `sm_75`, Ampere `sm_80`, Hopper `sm_90`, and GeForce
    # Blackwell `sm_120`; datacenter Blackwell `sm_100`/`sm_101` images are NOT
    # pinned and such devices need the explicit `IOM_CUDA_ARCHITECTURES`
    # override below. An image without the statements reports the leaf
    # `Unsupported`, never a fallback, and an explicit
    # `IOM_CUDA_ARCHITECTURES` overrides the list for an environment that needs
    # a different one.
    if(NOT DEFINED IOM_CUDA_ARCHITECTURES)
        set(IOM_CUDA_ARCHITECTURES "75;80;90")
        if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
            string(APPEND IOM_CUDA_ARCHITECTURES ";120")
        endif()
    endif()
    set_target_properties(iom_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${IOM_CUDA_ARCHITECTURES}"
    )

    target_include_directories(iom_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(iom_cuda
        PRIVATE
            libiom
            CUDA::cuda_driver
            CUDA::cudart
    )

endif()
if(ROCM_ENABLED)
    # Config mode discovers the configured SDK and exposes hip::host for the
    # backend target. Check the package version separately so newer major
    # ROCm releases remain valid while preserving the 7.2 minimum.
    find_package(hip CONFIG REQUIRED
        PATHS "${ROCM_PATH}"
        NO_DEFAULT_PATH
    )
    if(NOT DEFINED hip_VERSION OR hip_VERSION VERSION_LESS 7.2)
        message(FATAL_ERROR
            "ROCm 7.2 or newer is required for WMMA; detected ${hip_VERSION}"
        )
    endif()
    enable_language(HIP)

    add_library(iom_rocm STATIC
        src/rocm/device.cpp
        src/rocm/device_tensor.cpp
        src/rocm/copy.hip
        src/rocm/sdpa_softmax.hip
        src/rocm/sdpa_softmax.hpp
        src/rocm/sdpa_lowering.hpp
        src/rocm/copy.hpp
        src/rocm/sdpa_matrix.hip
        src/rocm/sdpa_matrix.hpp
        src/rocm/driver.cpp
        src/rocm/driver.hpp
        src/rocm/device_internal.hpp
        src/shared/metadata_slot_pool.hpp
        src/shared/event_ring.hpp
        src/shared/gpu_queue.hpp
        src/shared/gpu_queue_lifecycle.inl
        src/shared/gpu_queue_operations.inl
        src/shared/standard_tiled_copy.hpp
        src/shared/standard_tiled_embedding.hpp
        src/shared/standard_tiled_embedding.inl
        src/shared/standard_tiled_linear.inl
        src/shared/standard_tiled_copy.inl
        src/shared/standard_tiled_copy_metadata.inl
        src/shared/standard_tiled_rope.inl
        include/iom/rocm/device.hpp
    )

    target_include_directories(iom_rocm
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(iom_rocm
        PRIVATE
            hip::host
    )

    target_compile_definitions(iom_rocm
        PRIVATE
            __HIP_PLATFORM_AMD__
    )

    # Unoptimized HIP device images retain the runtime assertion/hostcall
    # support even for kernels that do not use it. APUs without PCIe atomics
    # reject those images at dispatch, so keep debug and unconfigured builds
    # debuggable while applying the minimum optimization that removes the
    # unreachable hostcall path.
    target_compile_options(iom_rocm
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>>:-O1>
    )
endif()
if(SYCL_ENABLED)
    add_library(iom_sycl STATIC
        src/sycl/device.cpp
        src/sycl/device_tensor.cpp
        src/sycl/copy.cpp
        src/sycl/embedding.cpp
        src/sycl/embedding.hpp
        src/sycl/queue_binary.cpp
        src/sycl/queue_embedding.cpp
        src/sycl/queue_rmsnorm.cpp
        src/sycl/queue_rope.cpp
        src/sycl/queue_linear.cpp
        src/sycl/queue_silu.cpp
        src/sycl/queue_sdpa.cpp
        src/sycl/queue.cpp
        src/sycl/add.cpp
        src/sycl/sdpa_matrix.cpp
        src/sycl/sdpa_matrix.hpp
        src/sycl/sdpa_nonmatrix.cpp
        src/sycl/sdpa_nonmatrix.hpp
        src/sycl/sdpa_stage.hpp
        src/sycl/runtime.hpp
        src/sycl/copy.hpp
        src/sycl/device_internal.hpp
        src/sycl/queue_support.hpp
        src/sycl/queue_internal.hpp
        include/iom/sycl/device.hpp
        src/shared/standard_tiled_copy.inl
        src/shared/standard_tiled_embedding.hpp
        src/shared/standard_tiled_embedding.inl
        src/shared/standard_tiled_copy_metadata.inl
    )

    target_include_directories(iom_sycl
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/sycl
            ${PROJECT_SOURCE_DIR}/src/shared
            ${IOM_SYCL_INCLUDE_DIR}
    )

    target_link_libraries(iom_sycl
        PRIVATE
            libiom
    )

    target_compile_options(iom_sycl
        PRIVATE
            -fsycl
            # The contract forbids flush-to-zero and fast-math, so the port's
            # own translation units keep IEEE subnormal and NaN/infinity
            # behavior instead of inheriting the compiler's fast default model.
            -ffp-model=precise
    )

    target_link_options(iom_sycl
        PRIVATE
            -fsycl
    )

endif()


add_executable(iom_generate
    tools/iom_generate.cpp
)

target_link_libraries(iom_generate
    PRIVATE
        libiom
)

if(CUDA_ENABLED)
    target_compile_definitions(iom_generate
        PRIVATE
            IOM_GENERATE_CUDA_ENABLED
    )
    target_link_libraries(iom_generate
        PRIVATE
            iom_cuda
            CUDA::cuda_driver
            CUDA::cudart
    )
endif()

if(ROCM_ENABLED)
    target_compile_definitions(iom_generate
        PRIVATE
            IOM_GENERATE_ROCM_ENABLED
            __HIP_PLATFORM_AMD__
    )
    target_link_libraries(iom_generate
        PRIVATE
            iom_rocm
            hip::host
    )
endif()

if(SYCL_ENABLED)
    target_compile_definitions(iom_generate
        PRIVATE
            IOM_GENERATE_SYCL_ENABLED
    )
    target_compile_options(iom_generate
        PRIVATE
            -fsycl
    )
    target_link_options(iom_generate
        PRIVATE
            -fsycl
    )
    target_link_libraries(iom_generate
        PRIVATE
            iom_sycl
    )
endif()


include(CTest)
if(BUILD_TESTING)
    if(CUDA_ENABLED)
        target_compile_definitions(iom_cuda PUBLIC IOM_ENABLE_TESTING)
    endif()
    if(SYCL_ENABLED)
        target_compile_definitions(iom_sycl PRIVATE IOM_ENABLE_TESTING)
    endif()
    if(ROCM_ENABLED)
        target_compile_definitions(iom_rocm PUBLIC IOM_ENABLE_TESTING)
    endif()

    # The AVX-512 BF16 stage/path observation seam is a test-build-only
    # contract: the definition is private to libiom and to the CPU test targets
    # that query it, so a production build contains neither the observation
    # functions nor their counter state. The detector itself is unaffected; it
    # is baseline code in every configuration.
    target_compile_definitions(libiom
        PRIVATE
            IOM_AVX512_BF16_TESTING
    )
endif()

if(BUILD_TESTING)
  find_package(doctest CONFIG QUIET)
  if(NOT doctest_FOUND)
    message(FATAL_ERROR
            "doctest was not found. On Ubuntu, install it with:\n"
            "  sudo apt update && sudo apt install doctest-dev\n"
            "Then rerun CMake."
        )
  endif()

  add_subdirectory(test)
endif()
