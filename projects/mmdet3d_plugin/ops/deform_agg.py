import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

torch.ops.load_library(os.path.join(current_dir, 'custom_ops.cpython-310-x86_64-linux-gnu.so'))


def deformable_aggregation_function(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
    dtype = mc_ms_feat.dtype
    mc_ms_feat = mc_ms_feat.to(torch.float32)
    spatial_shape = spatial_shape.to(torch.int32)
    scale_start_index = scale_start_index.to(torch.int32)
    sampling_location = sampling_location.to(torch.float32)
    weights = weights.to(torch.float32)
    result = torch.ops.custom_ops.deform_agg(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
    result = result.to(dtype)
    return result


@torch.library.impl_abstract("custom_ops::deform_agg")
def deform_agg_abstract(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
    return torch.empty([weights.shape[0], weights.shape[1],
                        mc_ms_feat.shape[2]]).to(mc_ms_feat.dtype).to(mc_ms_feat.device)
