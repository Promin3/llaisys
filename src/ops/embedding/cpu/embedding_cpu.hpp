#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>


namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const std::vector<size_t>& index_shape, const std::vector<size_t>& weight_shape);
}