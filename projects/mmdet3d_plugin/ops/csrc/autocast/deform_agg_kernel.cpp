#include "../deform_agg.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace custom_ops {
namespace ops {

namespace {

at::Tensor deform_agg_autocast(const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                               const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                               const at::Tensor &weights) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return deform_agg(at::autocast::cached_cast(at::kFloat, mc_ms_feat), spatial_shape, scale_start_index,
                    at::autocast::cached_cast(at::kFloat, sampling_location),
                    at::autocast::cached_cast(at::kFloat, weights))
      .to(mc_ms_feat.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(custom_ops, Autocast, m) {
  m.impl(TORCH_SELECTIVE_NAME("custom_ops::deform_agg"), TORCH_FN(deform_agg_autocast));
}

} // namespace ops
} // namespace custom_ops
