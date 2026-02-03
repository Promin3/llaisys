#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstring>
#include <vector>
using namespace llaisys::utils;

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, llaisysDataType_t type, const std::vector<size_t> &index_shape, const std::vector<size_t> &weight_shape) {
    size_t nrow = index_shape[0];
    size_t ncol = weight_shape[1];
    size_t row_size = ncol * dsize(type);
    for (size_t i = 0; i < nrow; i++) {
        memcpy(out + i * ncol, weight + index[i] * ncol, row_size);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const std::vector<size_t> &index_shape, const std::vector<size_t> &weight_shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), type, index_shape, weight_shape);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), type, index_shape, weight_shape);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), type, index_shape, weight_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
