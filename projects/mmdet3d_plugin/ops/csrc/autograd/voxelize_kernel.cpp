#include "../voxelize.h"

#include <iostream>
#include <torch/autograd.h>
#include <torch/types.h>
namespace custom_ops {
namespace ops {

namespace {

class VoxelizeFunction : public torch::autograd::Function<VoxelizeFunction> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx, const torch::autograd::Variable &points,
          const torch::autograd::Variable &voxel_size, const torch::autograd::Variable &coords_range,
          const torch::autograd::Variable &num_points, const int64_t max_points, const int64_t max_voxels,
          const int64_t grid_size_x, const int64_t grid_size_y) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output =
        voxelize(points, voxel_size, coords_range, num_points, max_points, max_voxels, grid_size_x, grid_size_y);

    ctx->save_for_backward({points, voxel_size, coords_range, num_points});

    return {
        std::get<0>(output),
        std::get<1>(output),
        std::get<2>(output),
        std::get<3>(output),
    };
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 const torch::autograd::variable_list &grad_output) {
    auto saved = ctx->get_saved_variables();
    auto points = saved[0];
    auto voxel_size = saved[1];
    auto coords_range = saved[2];
    auto num_points = saved[3];

    auto max_points = 0;
    auto max_voxels = 0;
    auto grid_size_x = 0;
    auto grid_size_y = 0;
    auto grad_points = detail::_voxelize_backward(grad_output[0], points, voxel_size, coords_range, num_points,
                                                  max_points, max_voxels, grid_size_x, grid_size_y);

    return {
        grad_points,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
    };
  }
};

// TODO: There should be an easier way to do this
class VoxelizeBackwardFunction : public torch::autograd::Function<VoxelizeBackwardFunction> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx, const torch::autograd::Variable &grad,
          const torch::autograd::Variable &points, const torch::autograd::Variable &voxel_size,
          const torch::autograd::Variable &coords_range, const torch::autograd::Variable &num_points,
          const int64_t max_points, const int64_t max_voxels, const int64_t grid_size_x, const int64_t grid_size_y) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto grad_points = detail::_voxelize_backward(grad, points, voxel_size, coords_range, num_points, max_points,
                                                  max_voxels, grid_size_x, grid_size_y);

    return {
        grad_points,
    };
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 const torch::autograd::variable_list &grad_output) {
    TORCH_CHECK(0, "double backwards on voxelize not supported");
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
voxelize_autograd(const at::Tensor &points, const at::Tensor &voxel_size, const at::Tensor &coords_range,
                  const at::Tensor &num_points, const int64_t max_points, const int64_t max_voxels,
                  const int64_t grid_size_x, const int64_t grid_size_y) {
  auto result = VoxelizeFunction::apply(points, voxel_size, coords_range, num_points, max_points, max_voxels,
                                        grid_size_x, grid_size_y);
  return std::make_tuple(result[0], result[1], result[2], result[3]);
}

at::Tensor voxelize_backward_autograd(const at::Tensor &grad, const at::Tensor &points, const at::Tensor &voxel_size,
                                      const at::Tensor &coords_range, const at::Tensor &num_points,
                                      const int64_t max_points, const int64_t max_voxels, const int64_t grid_size_x,
                                      const int64_t grid_size_y) {
  auto result = VoxelizeBackwardFunction::apply(grad, points, voxel_size, coords_range, num_points, max_points,
                                                max_voxels, grid_size_x, grid_size_y);

  return result[0];
}

} // namespace

TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("custom_ops::voxelize"), TORCH_FN(voxelize_autograd));
  m.impl(TORCH_SELECTIVE_NAME("custom_ops::_voxelize_backward"), TORCH_FN(voxelize_backward_autograd));
}

} // namespace ops
} // namespace custom_ops
