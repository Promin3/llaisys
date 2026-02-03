#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void rope_impl(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta) {
    using llaisys::utils::cast;

    const size_t half = head_dim / 2;

    for (size_t t = 0; t < seq_len; t++) {
        const float pos = static_cast<float>(pos_ids[t]);
        for (size_t h = 0; h < n_heads; h++) {
            const size_t base = t * n_heads * head_dim + h *head_dim;
            const T *x = in + base;
            T *y = out + base;
            for (size_t j = 0; j < half; j++) {
                const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
                const float denom = std::pow(theta, exponent);
                const float phi = pos / denom;
                const float s = std::sin(phi);
                const float c = std::cos(phi);

                const float a = cast<float>(x[j]);
                const float b = cast<float>(x[j + half]);

                const float a2 = a * c - b * s;
                const float b2 = b * c + a * s;

                y[j] = cast<T>(a2);
                y[j + half] = cast<T>(b2);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids,
            seq_len,
            n_heads,
            head_dim,
            theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids,
            seq_len,
            n_heads,
            head_dim,
            theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids,
            seq_len,
            n_heads,
            head_dim,
            theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
