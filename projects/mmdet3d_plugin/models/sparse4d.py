# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
from torch import Tensor

from typing import Dict, List, Optional

from mmdet3d.registry import MODELS
from mmdet3d.models import Base3DDetector
from mmdet3d.structures import Det3DDataSample
from .grid_mask import GridMask

import numpy as np

from .utils import feature_maps_format

__all__ = ["Sparse4D"]


@MODELS.register_module()
class Sparse4D(Base3DDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.head = MODELS.build(head)
        self.use_grid_mask = use_grid_mask
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
        if return_depth and self.depth_branch is not None:
            focals = [meta.metainfo.get("focal") for meta in metas]
            focals = torch.tensor(np.stack(focals, axis=0)).to(feature_maps[0].dtype)
            focals = focals.to(feature_maps[0].device) # TODO this should not happen here
            depths = self.depth_branch(feature_maps, focals)
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def _forward(self, batch_inputs: Tensor, batch_data_samples=None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def predict(
        self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ) -> List[Det3DDataSample]:
        img = batch_inputs_dict['img']
        img = torch.stack(img, axis=0)

        feature_maps = self.extract_feat(img)
        model_outs = self.head(feature_maps, batch_data_samples)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        if torch.onnx.is_in_onnx_export():
            boxes_3d, scores_3d, labels_3d, cls_scores, instance_ids = output[0]['img_bbox'].values()
            return boxes_3d, scores_3d, labels_3d, cls_scores, instance_ids
        return output

    def loss(
        self, batch_inputs_dict: Dict[str, Optional[torch.Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ):
        img = batch_inputs_dict['img']
        img = torch.stack(img, axis=0)

        feature_maps, depths = self.extract_feat(img, True, batch_data_samples)
        model_outs = self.head(feature_maps, batch_data_samples)
        output = self.head.loss(model_outs, batch_data_samples)
        if depths is not None and "gt_depth" in batch_data_samples[0].metainfo:
            gt_depths = []
            num_scales = len(batch_data_samples[0].metainfo['gt_depth'])
            for scale in range(num_scales):
                batch_depth = []
                for sample in batch_data_samples:
                    batch_depth.append(sample.metainfo['gt_depth'][scale])
                batch_depth = torch.concat(batch_depth, axis=0)
                batch_depth = batch_depth.to(depths[0].dtype).to(depths[0].device)
                gt_depths.append(batch_depth)
            output["loss_dense_depth"] = self.depth_branch.loss(depths, gt_depths)
        return output
