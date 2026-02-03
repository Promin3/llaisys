#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    ASSERT(out->shape()[0]==index->shape()[0],"dim 0 of out tensor should be equal to dim 0 of idx tensor");
    ASSERT(out->shape()[1]==weight->shape()[1],"dim 1 of out tensor should be equal to dim 1 of weight tensor");

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),index->data(),weight->data(),weight->dtype(),index->shape(),weight->shape());
    }
   
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(),index->data(),weight->data(),weight->dtype(),index->shape(),weight->shape());
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
