#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "cuda_helpers.h"

namespace custom_ops
{
  namespace ops
  {

    namespace
    {

      inline unsigned int GET_THREADS()
      {
#ifdef WITH_HIP
        return 256;
#endif
        return 512;
      }

      inline unsigned int GET_BLOCKS(const unsigned int THREADS, const int64_t N)
      {
        int64_t kMaxGridNum = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
        return (unsigned int)std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
      }

      __device__ float bilinear_sampling(
          const float *&bottom_data, const int &height, const int &width,
          const int &num_embeds, const float &h_im, const float &w_im,
          const int64_t &base_ptr)
      {
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;

        const float lh = h_im - h_low;
        const float lw = w_im - w_low;
        const float hh = 1 - lh, hw = 1 - lw;

        const int w_stride = num_embeds;
        const int h_stride = width * w_stride;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        float v1 = 0;
        if (h_low >= 0 && w_low >= 0)
        {
          const int64_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
          v1 = bottom_data[ptr1];
        }
        float v2 = 0;
        if (h_low >= 0 && w_high <= width - 1)
        {
          const int64_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
          v2 = bottom_data[ptr2];
        }
        float v3 = 0;
        if (h_high <= height - 1 && w_low >= 0)
        {
          const int64_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
          v3 = bottom_data[ptr3];
        }
        float v4 = 0;
        if (h_high <= height - 1 && w_high <= width - 1)
        {
          const int64_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
          v4 = bottom_data[ptr4];
        }

        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

        const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
        return val;
      }

      __device__ void bilinear_sampling_grad(
          const float *&bottom_data, const float &weight,
          const int &height, const int &width,
          const int &num_embeds, const float &h_im, const float &w_im,
          const int64_t &base_ptr,
          const float &grad_output,
          float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights)
      {
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;

        const float lh = h_im - h_low;
        const float lw = w_im - w_low;
        const float hh = 1 - lh, hw = 1 - lw;

        const int w_stride = num_embeds;
        const int h_stride = width * w_stride;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
        const float top_grad_mc_ms_feat = grad_output * weight;
        float grad_h_weight = 0, grad_w_weight = 0;

        float v1 = 0;
        if (h_low >= 0 && w_low >= 0)
        {
          const int64_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
          v1 = bottom_data[ptr1];
          grad_h_weight -= hw * v1;
          grad_w_weight -= hh * v1;
          atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
        }
        float v2 = 0;
        if (h_low >= 0 && w_high <= width - 1)
        {
          const int64_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
          v2 = bottom_data[ptr2];
          grad_h_weight -= lw * v2;
          grad_w_weight += hh * v2;
          atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
        }
        float v3 = 0;
        if (h_high <= height - 1 && w_low >= 0)
        {
          const int64_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
          v3 = bottom_data[ptr3];
          grad_h_weight += hw * v3;
          grad_w_weight -= lh * v3;
          atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
        }
        float v4 = 0;
        if (h_high <= height - 1 && w_high <= width - 1)
        {
          const int64_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
          v4 = bottom_data[ptr4];
          grad_h_weight += lw * v4;
          grad_w_weight += lh * v4;
          atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
        }

        const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
        atomicAdd(grad_weights, grad_output * val);
        atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
        atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
      }

      __global__ void deformable_aggregation_kernel(
          const int num_kernels,
          float *output,
          const float *mc_ms_feat,
          const int *spatial_shape,
          const int *scale_start_index,
          const float *sample_location,
          const float *weights,
          int batch_size,
          int num_cams,
          int num_feat,
          int num_embeds,
          int num_scale,
          int num_anchors,
          int num_pts,
          int num_groups)
      {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_kernels)
          return;

        const int channel_index = idx % num_embeds;
        idx /= num_embeds;

        const int group_idx = channel_index / (num_embeds / num_groups);

        int anchor_index = idx % num_anchors;
        idx /= num_anchors;

        const int batch_index = idx;

        anchor_index = batch_index * num_anchors + anchor_index;

        float outval = 0.f;
        for (int pts_index = 0; pts_index < num_pts; ++pts_index)
        {
          for (int cam_index = 0; cam_index < num_cams; ++cam_index)
          {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index)
            {
              const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

              const float loc_w = sample_location[loc_offset];
              if (loc_w <= 0 || loc_w >= 1)
                continue;
              const float loc_h = sample_location[loc_offset + 1];
              if (loc_h <= 0 || loc_h >= 1)
                continue;

              int cam_scale_index = cam_index * num_scale + scale_index;
              const int64_t value_offset =
                  (batch_index * num_feat + static_cast<int64_t>(scale_start_index[cam_scale_index])) * num_embeds + channel_index;

              cam_scale_index = cam_scale_index << 1;
              const int h = spatial_shape[cam_scale_index];
              const int w = spatial_shape[cam_scale_index + 1];

              const float h_im = loc_h * h - 0.5;
              const float w_im = loc_w * w - 0.5;

              const int weights_idx = (((anchor_index * num_pts + pts_index) * num_cams + cam_index) * num_scale + scale_index) * num_groups + group_idx;
              const float weight = *(weights + weights_idx);
              outval += bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight;
            }
          }
        }

        *(output + anchor_index * num_embeds + channel_index) = outval;
      }

      __global__ void deformable_aggregation_grad_kernel(
          const int num_kernels,
          const float *mc_ms_feat,
          const int *spatial_shape,
          const int *scale_start_index,
          const float *sample_location,
          const float *weights,
          const float *grad_output,
          float *grad_mc_ms_feat,
          float *grad_sampling_location,
          float *grad_weights,
          int batch_size,
          int num_cams,
          int num_feat,
          int num_embeds,
          int num_scale,
          int num_anchors,
          int num_pts,
          int num_groups)
      {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_kernels)
          return;

        const int weights_ptr = idx / (num_embeds / num_groups);
        const int channel_index = idx % num_embeds;
        idx /= num_embeds;
        const int scale_index = idx % num_scale;
        idx /= num_scale;

        const int cam_index = idx % num_cams;
        idx /= num_cams;
        const int pts_index = idx % num_pts;
        idx /= num_pts;

        int anchor_index = idx % num_anchors;
        idx /= num_anchors;
        const int batch_index = idx % batch_size;
        idx /= batch_size;

        anchor_index = batch_index * num_anchors + anchor_index;
        const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

        const float loc_w = sample_location[loc_offset];
        if (loc_w <= 0 || loc_w >= 1)
          return;
        const float loc_h = sample_location[loc_offset + 1];
        if (loc_h <= 0 || loc_h >= 1)
          return;

        const float grad = grad_output[anchor_index * num_embeds + channel_index];

        int cam_scale_index = cam_index * num_scale + scale_index;
        const int64_t value_offset = (batch_index * num_feat + static_cast<int64_t>(scale_start_index[cam_scale_index])) * num_embeds + channel_index;

        cam_scale_index = cam_scale_index << 1;
        const int h = spatial_shape[cam_scale_index];
        const int w = spatial_shape[cam_scale_index + 1];

        const float h_im = loc_h * h - 0.5;
        const float w_im = loc_w * w - 0.5;

        /* atomicAdd( */
        /*     output + anchor_index * num_embeds + channel_index, */
        /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
        /* ); */
        const float weight = weights[weights_ptr];
        float *grad_weights_ptr = grad_weights + weights_ptr;
        float *grad_location_ptr = grad_sampling_location + loc_offset;
        bilinear_sampling_grad(
            mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
            value_offset,
            grad,
            grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr);
      }

      at::Tensor deform_agg_forward_kernel(
          const at::Tensor &mc_ms_feat,
          const at::Tensor &spatial_shape,
          const at::Tensor &scale_start_index,
          const at::Tensor &sampling_location,
          const at::Tensor &weights)
      {
        at::Tensor mc_ms_feat_c = mc_ms_feat.contiguous();
        at::Tensor spatial_shape_c = spatial_shape.contiguous();
        at::Tensor scale_start_index_c = scale_start_index.contiguous();
        at::Tensor sampling_location_c = sampling_location.contiguous();
        at::Tensor weights_c = weights.contiguous();

        TORCH_CHECK(mc_ms_feat_c.ndimension() == 3);
        TORCH_CHECK(spatial_shape_c.ndimension() == 3);
        TORCH_CHECK(scale_start_index_c.ndimension() == 2);
        TORCH_CHECK(sampling_location_c.ndimension() == 5);
        TORCH_CHECK(weights_c.ndimension() == 6);
        TORCH_CHECK(mc_ms_feat_c.is_cuda(), "input must be a CUDA tensor");

        at::DeviceGuard guard(mc_ms_feat_c.device());

        int batch_size = mc_ms_feat_c.size(0);
        int num_feat = mc_ms_feat_c.size(1);
        int num_embeds = mc_ms_feat_c.size(2);
        int num_cams = spatial_shape_c.size(0);
        int num_scale = spatial_shape_c.size(1);
        int num_anchors = sampling_location_c.size(1);
        int num_pts = sampling_location_c.size(2);
        int num_groups = weights_c.size(5);

        auto output = at::zeros({batch_size, num_anchors, num_embeds}, mc_ms_feat_c.options());
        if (batch_size == 0)
        {
          return output;
        }

        const int64_t num_kernels = batch_size * num_embeds * num_anchors;

        const unsigned int threads = GET_THREADS();
        const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

        // AT_DISPATCH_FLOATING_TYPES(
        //     mc_ms_feat_c.scalar_type(), "deformable_aggregation_kernel", ([&] {
        deformable_aggregation_kernel<<<blocks, threads>>>(
            num_kernels,
            output.data_ptr<float>(),
            mc_ms_feat_c.data_ptr<float>(),
            spatial_shape_c.data_ptr<int>(),
            scale_start_index_c.data_ptr<int>(),
            sampling_location_c.data_ptr<float>(),
            weights_c.data_ptr<float>(),
            batch_size,
            num_cams,
            num_feat,
            num_embeds,
            num_scale,
            num_anchors,
            num_pts,
            num_groups);
        //    }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
      }

      std::tuple<at::Tensor, at::Tensor, at::Tensor>
      deform_agg_backward_kernel(
          const at::Tensor &grad_out,
          const at::Tensor &mc_ms_feat,
          const at::Tensor &spatial_shape,
          const at::Tensor &scale_start_index,
          const at::Tensor &sampling_location,
          const at::Tensor &weights)
      {
        at::Tensor grad_out_c = grad_out.contiguous();
        at::Tensor mc_ms_feat_c = mc_ms_feat.contiguous();
        at::Tensor spatial_shape_c = spatial_shape.contiguous();
        at::Tensor scale_start_index_c = scale_start_index.contiguous();
        at::Tensor sampling_location_c = sampling_location.contiguous();
        at::Tensor weights_c = weights.contiguous();

        TORCH_CHECK(mc_ms_feat_c.ndimension() == 3);
        TORCH_CHECK(spatial_shape_c.ndimension() == 3);
        TORCH_CHECK(scale_start_index_c.ndimension() == 2);
        TORCH_CHECK(sampling_location_c.ndimension() == 5);
        TORCH_CHECK(weights_c.ndimension() == 6);
        TORCH_CHECK(mc_ms_feat_c.is_cuda(), "input must be a CUDA tensor");

        at::DeviceGuard guard(mc_ms_feat_c.device());

        int batch_size = mc_ms_feat_c.size(0);
        int num_feat = mc_ms_feat_c.size(1);
        int num_embeds = mc_ms_feat_c.size(2);
        int num_cams = spatial_shape_c.size(0);
        int num_scale = spatial_shape_c.size(1);
        int num_anchors = sampling_location_c.size(1);
        int num_pts = sampling_location_c.size(2);
        int num_groups = weights_c.size(5);

        auto grad_mc_ms_feat = at::zeros_like(mc_ms_feat_c);
        auto grad_sampling_location = at::zeros_like(sampling_location_c);
        auto grad_weights = at::zeros_like(weights_c);
        if (batch_size == 0)
        {
          return std::make_tuple(grad_mc_ms_feat, grad_sampling_location, grad_weights);
        }

        const int64_t num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

        const unsigned int threads = GET_THREADS();
        const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

        // AT_DISPATCH_FLOATING_TYPES(
        //     mc_ms_feat_c.scalar_type(), "deformable_aggregation_grad_kernel", ([&] {
        deformable_aggregation_grad_kernel<<<blocks, threads>>>(
            num_kernels,
            mc_ms_feat_c.data_ptr<float>(),
            spatial_shape_c.data_ptr<int>(),
            scale_start_index_c.data_ptr<int>(),
            sampling_location_c.data_ptr<float>(),
            weights_c.data_ptr<float>(),
            grad_out_c.data_ptr<float>(),
            grad_mc_ms_feat.data_ptr<float>(),
            grad_sampling_location.data_ptr<float>(),
            grad_weights.data_ptr<float>(),
            batch_size,
            num_cams,
            num_feat,
            num_embeds,
            num_scale,
            num_anchors,
            num_pts,
            num_groups);
        //    }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(grad_mc_ms_feat, grad_sampling_location, grad_weights);
      }

    } // namespace

    TORCH_LIBRARY_IMPL(custom_ops, CUDA, m)
    {
      m.impl(
          TORCH_SELECTIVE_NAME("custom_ops::deform_agg"),
          TORCH_FN(deform_agg_forward_kernel));
      m.impl(
          TORCH_SELECTIVE_NAME("custom_ops::_deform_agg_backward"),
          TORCH_FN(deform_agg_backward_kernel));
    }

  } // namespace ops
} // namespace custom_ops
