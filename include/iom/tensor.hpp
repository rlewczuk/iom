#pragma once

#include <vector>

namespace iom {

    enum class TileSize {

    };

    enum class DataType {
        BOOL,

        I2, U2,
        I4, U4,
        I8, U8,
        I16, U16,
        I32, U32,
        I64, U64,

        F4_E2M1,
        F6_E2M3,
        F6_E3M2,

        F8_E4M3FN,
        F8_E5M2,
        F8_E8M0,

        F16,
        BF16,
        F32,
        F64,
    };

    enum class QuantizationFormat {
        // No data grouping, just plain numbers (of appropriate DataType)
        NONE,

        // Generic formats, if you actually implement them
        INT8_SYMMETRIC,
        INT8_ASYMMETRIC,
        INT4_SYMMETRIC,
        INT4_ASYMMETRIC,

        // OCP
        OCP_MXFP4,
        OCP_MXFP8_E4M3,
        OCP_MXFP8_E5M2,

        // NVIDIA
        NVIDIA_NVFP4,

        // GGML
        GGML_Q4_0,
        GGML_Q4_1,
        GGML_Q5_0,
        GGML_Q5_1,
        GGML_Q8_0,
        GGML_Q2_K,
        GGML_Q3_K,
        GGML_Q4_K,
        GGML_Q5_K,
        GGML_Q6_K,
        // ...

        // Tenstorrent
        TT_BFP2,
        TT_BFP2A,
        TT_BFP4,
        TT_BFP4A,
        TT_BFP8,
        TT_BFP8A,
    };

    class TensorShape {
        std::vector<int> shape;
    };

    /**
     * Represents a tensor.
     */
    class Tensor {
        // TBD private fields here
    public:

        virtual ~Tensor() = default;

    };

}
