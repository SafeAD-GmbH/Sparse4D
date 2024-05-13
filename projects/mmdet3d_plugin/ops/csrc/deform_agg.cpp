#include "deform_agg.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace custom_ops {
namespace ops {

at::Tensor deform_agg(const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                      const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                      const at::Tensor &weights) {
  C10_LOG_API_USAGE_ONCE("custom_ops.csrc.ops.deform_agg.deformdeform_agg_conv2d");
  static auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("custom_ops::deform_agg", "").typed<decltype(deform_agg)>();
  return op.call(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights);
}

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_deform_agg_backward(const at::Tensor &grad, const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                     const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                     const at::Tensor &weights) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("custom_ops::_deform_agg_backward", "")
                       .typed<decltype(_deform_agg_backward)>();
  return op.call(grad, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(custom_ops, m) {
  m.impl_abstract_pystub("projects.mmdet3d_plugin.ops.deform_agg");
  m.def(TORCH_SELECTIVE_SCHEMA("custom_ops::deform_agg(Tensor mc_ms_feat, Tensor "
                               "spatial_shape, Tensor scale_start_index, Tensor "
                               "sampling_location, Tensor weights) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA("custom_ops::_deform_agg_backward(Tensor grad, Tensor mc_ms_feat, Tensor "
                               "spatial_shape, Tensor scale_start_index, Tensor sampling_location, "
                               "Tensor weights) -> (Tensor, Tensor, Tensor)"));
}

} // namespace ops
} // namespace custom_ops
