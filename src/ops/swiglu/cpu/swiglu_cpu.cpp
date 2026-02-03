#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void swiglu_impl(T *out, const T *gate, const T *up, size_t numel) {
    using llaisys::utils::cast;

    for (size_t i = 0; i < numel; i++) {
        const float g = cast<float>(gate[i]);
        const float u = cast<float>(up[i]);
        const float s = g / (1.0f + std::exp(-g));
        out[i] = cast<T>(u * s);
    }
}

} // namespace

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up),
            numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
