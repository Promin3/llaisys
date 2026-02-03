#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "ROPE: out/in must be 3D tensors");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "ROPE: pos_ids must be 1D tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "ROPE: pos_ids dtype must be int64");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "ROPE: all tensors must be contiguous.");

    const size_t seq_len = in->shape()[0];
    const size_t n_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];

    CHECK_ARGUMENT(pos_ids->shape()[0] == seq_len, "ROPE: pos_ids length must equal seq_len");
    CHECK_ARGUMENT(head_dim % 2 == 0, "ROPE: head_dim must be even");
    CHECK_ARGUMENT(theta > 0.0f, "ROPE: theta must be positive");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            theta);
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
