#include "embedding.hpp"

#include <sycl/sycl.hpp>

#include "../shared/standard_tiled_embedding.hpp"
#include "iom/iom.hpp"

namespace iom::detail {

inline void atomicOr(
        std::uint32_t* address, std::uint32_t value) {
    using Atomic = sycl::atomic_ref<
            std::uint32_t, sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>;
    Atomic(*address).fetch_or(value);
}

}  // namespace iom::detail

#define IOM_GPU_DEVICE inline
#define IOM_GPU_GLOBAL inline
#define IOM_GPU_GLOBAL_INDEX 0
#define IOM_GPU_GLOBAL_STRIDE 1
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) ((void)0)
#include "../shared/standard_tiled_embedding.inl"
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_STRIDE
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

// Force instantiation of the metadata writer for the SYCL queue's request
// type so this TU emits the symbol consumed by queue_embedding.cpp.
template iom::detail::EmbeddingMetadata
iom::detail::write_embedding_metadata<iom::DeviceOps::EmbeddingRequest>(
        void* host_storage, const void* device_storage,
        const iom::DeviceOps::EmbeddingRequest& request);

namespace iom::sycl_detail {

sycl::event launch_embedding_words(
        sycl::queue& queue, const sycl::event& reset,
        const unsigned char* table, const unsigned char* indices,
        unsigned char* output, std::uint32_t* status,
        const detail::EmbeddingMetadata& metadata) {
    const std::uint64_t total_words =
            metadata.header.plane_count * metadata.header.words_per_plane;
    return queue.submit([&](sycl::handler& handler) {
        handler.depends_on(reset);
        handler.parallel_for(
                sycl::range<1>(static_cast<std::size_t>(total_words)),
                [=](sycl::id<1> item) {
                    detail::embedding_one_word(
                            table, indices, output, status, metadata,
                            static_cast<std::uint64_t>(item[0]));
                });
    });
}

}  // namespace iom::sycl_detail
