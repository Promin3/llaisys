#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename T>
void self_attention_impl(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t qlen,
    size_t nh,
    size_t d,
    size_t kvlen,
    size_t nkvh,
    size_t dv,
    float scale) {
    using llaisys::utils::cast;

    const size_t group = nh / nkvh; // require divisible, checked by caller

    std::vector<float> logits(kvlen);

    // causal mask offset: allow attending to keys up to index t + (kvlen - qlen)
    const ptrdiff_t offset = static_cast<ptrdiff_t>(kvlen) - static_cast<ptrdiff_t>(qlen);

    for (size_t t = 0; t < qlen; t++) {
        const ptrdiff_t limit_i = static_cast<ptrdiff_t>(t) + offset;
        const size_t limit = limit_i < 0 ? 0 : static_cast<size_t>(std::min<ptrdiff_t>(limit_i, static_cast<ptrdiff_t>(kvlen - 1)));

        for (size_t h = 0; h < nh; h++) {
            const size_t kvh = h / group;

            const T *qvec = q + (t * nh + h) * d;
            T *out = attn_val + (t * nh + h) * dv;

            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < kvlen; s++) {
                if (s > limit) { 
                    logits[s] = -std::numeric_limits<float>::infinity();
                    continue;  
                }

                const T *kvec = k + (s * nkvh + kvh) * d;
                float dot = 0.0f;
                for (size_t i = 0; i < d; i++) {
                    dot += cast<float>(qvec[i]) * cast<float>(kvec[i]);
                }
                const float logit = dot * scale;
                logits[s] = logit;
                if (logit > max_logit) {
                    max_logit = logit;
                }
            }

            float denom = 0.0f;
            for (size_t s = 0; s < kvlen; s++) {
                if (!std::isfinite(logits[s])) {
                    logits[s] = 0.0f;
                    continue;
                }
                const float e = std::exp(logits[s] - max_logit);
                logits[s] = e;
                denom += e;
            }
            const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

            for (size_t j = 0; j < dv; j++) {
                float acc = 0.0f;
                for (size_t s = 0; s < kvlen; s++) {
                    const float w = logits[s] * inv_denom;
                    if (w == 0.0f) {
                        continue;
                    }
                    const T *vvec = v + (s * nkvh + kvh) * dv;
                    acc += w * cast<float>(vvec[j]);
                }
                out[j] = cast<T>(acc);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t qlen,
    size_t nh,
    size_t d,
    size_t kvlen,
    size_t nkvh,
    size_t dv,
    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            qlen,
            nh,
            d,
            kvlen,
            nkvh,
            dv,
            scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            qlen,
            nh,
            d,
            kvlen,
            nkvh,
            dv,
            scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            qlen,
            nh,
            d,
            kvlen,
            nkvh,
            dv,
            scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
