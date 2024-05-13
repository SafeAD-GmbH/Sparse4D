#pragma once

#include <ATen/ATen.h>

namespace custom_ops {
namespace ops {

at::Tensor deform_agg(const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                      const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                      const at::Tensor &weights);

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_deform_agg_backward(const at::Tensor &grad, const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                     const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                     const at::Tensor &weights);

} // namespace detail

} // namespace ops
} // namespace custom_ops
