#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "SelfAttention: all tensors must be 3D");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t d = q->shape()[2];

    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    const size_t kd = k->shape()[2];

    const size_t vlen = v->shape()[0];
    const size_t vkvh = v->shape()[1];
    const size_t dv = v->shape()[2];

    CHECK_ARGUMENT(kd == d, "SelfAttention: q/k head dim mismatch");
    CHECK_ARGUMENT(vlen == kvlen && vkvh == nkvh, "SelfAttention: k/v shape mismatch");
    CHECK_ARGUMENT(nkvh > 0 && nh > 0, "SelfAttention: head count must be > 0");
    CHECK_ARGUMENT(nh % nkvh == 0, "SelfAttention: nh must be divisible by nkvh (GQA grouping)");

    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == dv,
                   "SelfAttention: attn_val shape mismatch");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen, nh, d, kvlen, nkvh, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen, nh, d, kvlen, nkvh, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
