struct CopyMetadataHeader {
    std::uint64_t source_plane_offset;
    std::uint64_t destination_plane_offset;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint32_t bits;
    std::uint32_t leading_rank;
};
inline constexpr std::size_t kInlineMetadataMaxRank = 8;

struct InlineCopyMetadata {
    CopyMetadataHeader header;
    std::uint64_t values[3 * kInlineMetadataMaxRank];
};
static_assert(std::is_trivially_copyable_v<InlineCopyMetadata>);
static_assert(sizeof(InlineCopyMetadata) == 240);
static_assert(sizeof(InlineCopyMetadata) % 16 == 0);
// Fixed-slot layout contract: the compiled rank-eight copy descriptor is
// 48 header bytes plus 3 * kInlineMetadataMaxRank stride words, and the
// rank-eight descriptor a submission actually writes is
// 48 + 24 * (max rank - 2) leading-rank bytes. Both the compiled
// representation and the rank-eight payload must fit one 512-byte metadata
// slot (alignment 32). A future overflow requires an intentional ABI/spec
// update, never automatic slot growth.
static_assert(
        sizeof(CopyMetadataHeader)
                + 3 * (kInlineMetadataMaxRank - 2) * sizeof(std::uint64_t)
        == 48 + 24 * 6);
static_assert(
        sizeof(CopyMetadataHeader)
                + 3 * (kInlineMetadataMaxRank - 2) * sizeof(std::uint64_t)
        <= kMetadataSlotBytes);
static_assert(sizeof(InlineCopyMetadata) <= kMetadataSlotBytes);
static_assert(
        alignof(InlineCopyMetadata) <= 32
        && kMetadataSlotBytes % alignof(InlineCopyMetadata) == 0);
static_assert(sizeof(InlineCopyMetadata) <= 1024);

static_assert(std::is_trivially_copyable_v<CopyMetadataHeader>);

[[nodiscard]] std::size_t checked_metadata_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_metadata_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::size_t padded_dimension(std::size_t value) {
    const std::size_t tiles =
            value / TensorSpec::TILE + (value % TensorSpec::TILE != 0);
    return checked_metadata_mul(
            tiles, TensorSpec::TILE, "metadata size overflows");
}

struct CopyMetadataLayout {
    std::size_t bytes;
    std::size_t total_words;
};

template <typename View>
[[nodiscard]] CopyMetadataLayout copy_metadata_layout(
        const View& source, const View& destination) {
    const std::span<const std::size_t> dimensions =
            source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    if (leading_rank > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("metadata leading rank overflows");
    }
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = checked_metadata_mul(
                plane_count, dimensions[axis], "metadata plane count overflows");
    }
    const std::size_t padded_elements = checked_metadata_mul(
            padded_dimension(rows), padded_dimension(columns),
            "metadata element count overflows");
    const std::size_t plane_bits = checked_metadata_mul(
            padded_elements, leaf_bits(source.spec().data_type),
            "metadata plane bits overflows");
    const std::size_t words_per_plane = checked_metadata_add(
            plane_bits, 31, "metadata word count overflows")
            / 32;
    const std::size_t total_words_size = checked_metadata_mul(
            plane_count, words_per_plane, "metadata total words overflows");
    const std::size_t array_count = checked_metadata_mul(
            leading_rank, 3, "metadata array count overflows");
    const std::size_t array_bytes = checked_metadata_mul(
            array_count, sizeof(std::uint64_t),
            "metadata array bytes overflows");
    const std::size_t bytes = checked_metadata_add(
            sizeof(CopyMetadataHeader), array_bytes,
            "metadata allocation size overflows");
    (void)metadata_u64(rows, "metadata rows overflows");
    (void)metadata_u64(columns, "metadata columns overflows");
    (void)metadata_u64(plane_count, "metadata plane count overflows");
    (void)metadata_u64(total_words_size, "metadata total words overflows");
    for (const std::size_t stride : source.plane_strides()) {
        (void)metadata_u64(stride, "metadata source stride overflows");
    }
    for (const std::size_t stride : destination.plane_strides()) {
        (void)metadata_u64(stride, "metadata destination stride overflows");
    }
    (void)metadata_u64(
            source.plane_offset(), "metadata source offset overflows");
    (void)metadata_u64(
            destination.plane_offset(), "metadata destination offset overflows");
    return {bytes, total_words_size};
}

template <typename View>
void write_copy_metadata(
        std::byte* storage, const View& source,
        const View& destination) {
    const std::span<const std::size_t> dimensions =
            source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = checked_metadata_mul(
                plane_count, dimensions[axis], "metadata plane count overflows");
    }
    auto* header = reinterpret_cast<CopyMetadataHeader*>(storage);
    header->source_plane_offset = metadata_u64(
            source.plane_offset(), "metadata source offset overflows");
    header->destination_plane_offset = metadata_u64(
            destination.plane_offset(), "metadata destination offset overflows");
    header->rows = metadata_u64(rows, "metadata rows overflows");
    header->columns = metadata_u64(columns, "metadata columns overflows");
    header->plane_count = metadata_u64(
            plane_count, "metadata plane count overflows");
    header->bits = static_cast<std::uint32_t>(
            leaf_bits(source.spec().data_type));
    header->leading_rank = static_cast<std::uint32_t>(leading_rank);
    auto* values = reinterpret_cast<std::uint64_t*>(header + 1);
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        values[axis] = metadata_u64(
                source.plane_strides()[axis],
                "metadata source stride overflows");
        values[leading_rank + axis] = metadata_u64(
                destination.plane_strides()[axis],
                "metadata destination stride overflows");
        values[2 * leading_rank + axis] = metadata_u64(
                dimensions[axis], "metadata leading dimension overflows");
    }
}
template <typename View>
void write_copy_metadata(
        InlineCopyMetadata& storage, const View& source,
        const View& destination) {
    write_copy_metadata(
            reinterpret_cast<std::byte*>(&storage), source, destination);
}


IOM_GPU_DEVICE void copy_one_tiled_word(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader& metadata,
        const std::uint64_t* values, std::uint64_t word) {
    const std::uint64_t padded_rows =
            (metadata.rows / TensorSpec::TILE
             + (metadata.rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata.columns / TensorSpec::TILE
             + (metadata.columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata.bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t logical_plane = word / words_per_plane;
    const std::uint64_t word_in_plane = word % words_per_plane;
    const std::uint64_t* source_strides = values;
    const std::uint64_t* destination_strides =
            source_strides + metadata.leading_rank;
    const std::uint64_t* leading_dimensions =
            destination_strides + metadata.leading_rank;
    std::uint64_t source_plane = metadata.source_plane_offset;
    std::uint64_t destination_plane =
            metadata.destination_plane_offset;
    std::uint64_t rest = logical_plane;
    for (std::uint32_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % leading_dimensions[axis];
        rest /= leading_dimensions[axis];
        source_plane += coordinate * source_strides[axis];
        destination_plane += coordinate * destination_strides[axis];
    }
    copy_tiled_to_tiled_word(
            source, destination, source_plane, destination_plane,
            word_in_plane, metadata.rows, metadata.columns,
            metadata.bits);
}

IOM_GPU_DEVICE void grid_stride_copy_body(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader& metadata,
        const std::uint64_t* values) {
    const std::uint64_t padded_rows =
            (metadata.rows / TensorSpec::TILE
             + (metadata.rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata.columns / TensorSpec::TILE
             + (metadata.columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata.bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t total_words =
            metadata.plane_count * words_per_plane;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;

    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX; word < total_words;
         word += stride) {
        copy_one_tiled_word(source, destination, metadata, values, word);
    }
}

IOM_GPU_GLOBAL void grid_stride_copy_kernel(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader* metadata) {
    grid_stride_copy_body(
            source, destination, *metadata,
            reinterpret_cast<const std::uint64_t*>(metadata + 1));
}

IOM_GPU_GLOBAL void grid_stride_copy_inline_kernel(
        const unsigned char* source, unsigned char* destination,
        InlineCopyMetadata metadata) {
    grid_stride_copy_body(
            source, destination, metadata.header, metadata.values);
}

template <typename Policy>
void launch_grid_stride_copy(
        typename Policy::stream_type stream, const unsigned char* source,
        unsigned char* destination, const CopyMetadataHeader* metadata,
        std::size_t total_words) {
    const std::size_t launch_words = checked_metadata_add(
            total_words, 255, "metadata launch count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(launch_words / kThreads, kMaxBlocks));
    IOM_LAUNCH_KERNEL(
            grid_stride_copy_kernel, blocks, kThreads, stream,
            source, destination, metadata);
}

template <typename Policy>
void launch_grid_stride_copy(
        typename Policy::stream_type stream, const unsigned char* source,
        unsigned char* destination, const InlineCopyMetadata& metadata,
        std::size_t total_words) {
    const std::size_t launch_words = checked_metadata_add(
            total_words, 255, "metadata launch count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(launch_words / kThreads, kMaxBlocks));
    IOM_LAUNCH_KERNEL(
            grid_stride_copy_inline_kernel, blocks, kThreads, stream,
            source, destination, metadata);
}
