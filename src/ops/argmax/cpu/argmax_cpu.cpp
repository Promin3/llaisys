#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(size_t *max_idx, T *max_val, const T *vals, size_t val_size) {
    size_t max_i = 0;
    float max_f = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < val_size; ++i) {
        const float v = llaisys::utils::cast<float>(vals[i]);
        if (v > max_f) {
            max_f = v;
            max_i = i;
        }
    }
    *max_idx = max_i;
    *max_val = vals[max_i];
}


namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t vals_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), vals_size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), vals_size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), vals_size);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu

