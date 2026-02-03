#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void rms_norm_impl(
    T *out,
    const T *in,
    const T *weight,
    size_t m,
    size_t d,
    float eps) {
    using llaisys::utils::cast;

    for (size_t i = 0; i < m; i++) {
        const T *in_row = in + i * d;
        T *out_row = out + i * d;

        float mean_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            const float x = cast<float>(in_row[j]);
            mean_sq += x * x;
        }
        mean_sq /= static_cast<float>(d);
        mean_sq += eps;
        const float inv_rms = 1.0f / std::sqrt(mean_sq);

        for (size_t j = 0; j < d; j++) {
            const float x = cast<float>(in_row[j]);
            const float w = cast<float>(weight[j]);
            out_row[j] = cast<T>(x * inv_rms * w);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t m,
    size_t d,
    float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            m,
            d,
            eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            m,
            d,
            eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            m,
            d,
            eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
