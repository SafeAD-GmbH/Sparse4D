#include "../deform_agg.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace custom_ops {
namespace ops {

namespace {

class DeformAggFunction : public torch::autograd::Function<DeformAggFunction> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx, const torch::autograd::Variable &mc_ms_feat,
          const torch::autograd::Variable &spatial_shape, const torch::autograd::Variable &scale_start_index,
          const torch::autograd::Variable &sampling_location, const torch::autograd::Variable &weights) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output = deform_agg(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights);

    ctx->save_for_backward({mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights});

    return {
        output,
    };
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 const torch::autograd::variable_list &grad_output) {
    auto saved = ctx->get_saved_variables();
    auto mc_ms_feat = saved[0];
    auto spatial_shape = saved[1];
    auto scale_start_index = saved[2];
    auto sampling_location = saved[3];
    auto weights = saved[4];

    auto grads = detail::_deform_agg_backward(grad_output[0], mc_ms_feat, spatial_shape, scale_start_index,
                                              sampling_location, weights);
    auto grad_mc_ms_feat = std::get<0>(grads);
    auto grad_sampling_location = std::get<1>(grads);
    auto grad_weight = std::get<2>(grads);

    return {
        grad_mc_ms_feat, torch::autograd::Variable(), torch::autograd::Variable(), grad_sampling_location, grad_weight,
    };
  }
};

// TODO: There should be an easier way to do this
class DeformAggBackwardFunction : public torch::autograd::Function<DeformAggBackwardFunction> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx, const torch::autograd::Variable &grad,
          const torch::autograd::Variable &mc_ms_feat, const torch::autograd::Variable &spatial_shape,
          const torch::autograd::Variable &scale_start_index, const torch::autograd::Variable &sampling_location,
          const torch::autograd::Variable &weights) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto grads =
        detail::_deform_agg_backward(grad, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights);

    auto grad_mc_ms_feat = std::get<0>(grads);
    auto grad_sampling_location = std::get<1>(grads);
    auto grad_weight = std::get<2>(grads);

    return {
        grad_mc_ms_feat,
        grad_sampling_location,
        grad_weight,
    };
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 const torch::autograd::variable_list &grad_output) {
    TORCH_CHECK(0, "double backwards on deform_agg not supported");
  }
};

at::Tensor deform_agg_autograd(const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                               const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                               const at::Tensor &weights) {
  return DeformAggFunction::apply(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)[0];
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
deform_agg_backward_autograd(const at::Tensor &grad, const at::Tensor &mc_ms_feat, const at::Tensor &spatial_shape,
                             const at::Tensor &scale_start_index, const at::Tensor &sampling_location,
                             const at::Tensor &weights) {
  auto result =
      DeformAggBackwardFunction::apply(grad, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights);

  return std::make_tuple(result[0], result[1], result[2]);
}

} // namespace

TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("custom_ops::deform_agg"), TORCH_FN(deform_agg_autograd));
  m.impl(TORCH_SELECTIVE_NAME("custom_ops::_deform_agg_backward"), TORCH_FN(deform_agg_backward_autograd));
}

} // namespace ops
} // namespace custom_ops
