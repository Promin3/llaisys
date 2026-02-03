#include "linear_cpu.hpp"

#include "../../../utils.hpp"


template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k) {
    using llaisys::utils::cast;

    for (size_t i = 0; i < m; i++) {
        const T *in_row = in + i * k;
        T *out_row = out + i * n;
        for (size_t j = 0; j < n; j++) {
            const T *w_row = weight + j * k;
            float acc = 0.0f;
            for (size_t p = 0; p < k; p++) {
                acc += cast<float>(in_row[p]) * cast<float>(w_row[p]);
            }
            acc += cast<float>(bias[j]);
            out_row[j] = cast<T>(acc);
        }
    }
}


namespace llaisys::ops::cpu {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            m,
            n,
            k);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            m,
            n,
            k);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            m,
            n,
            k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
