import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


from rich import print

# intrinsics for undistort
INTRINSICS = {
    'camera_f': [862.2973400197181, 862.2899467578039, 908.0102434493892, 555.6826562646286],
    'camera_fz': [1875.4171534571185, 1875.8796135339023, 944.4637991524326, 540.8183446721481],
    'camera_fl': [861.2739379367756, 861.0034059931078, 894.1353575442253, 561.0563927861167],
    'camera_fr': [856.1734163825976, 856.6266628662479, 898.3465897039454, 530.0581693215881],
    'camera_bl': [863.1573708011513, 863.0565044704765, 891.745611313655, 570.26309487825],
    'camera_br': [865.8714472392189, 865.9430051991653, 905.2123978277147, 559.62160190928],
    'camera_b': [863.320549892077, 863.9751990627854, 998.7808536957959, 562.1776010693202]
}

@DATASETS.register_module()
class SafeAD3DDetTrackDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        print(f"[DATASET] Name of annotation file is: {self.ann_file}")

        if classes is not None:
            self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def __len__(self):
        return len(self.data_infos[:1100])

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        # 1. [DATA-DIFF] Adapted the collection of sequences according to our dataset
        curr_sequence = -1
        curr_token = None
        for idx in range(len(self.data_infos)):
            if self.data_infos[idx]['scene_token'] != curr_token:
                # new sequence
                curr_sequence += 1
                curr_token = self.data_infos[idx]['scene_token']
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            # 2. [DATA-DIFF] NANO SEC in safead dataset
            timestamp=info["timestamp"] / 1e9,
            # 3. [DATA-DIFF] Get constants for lidar2ego transformation
            lidar2ego_rotation=[0.99972822181831, -0.016481156718805186, -0.014689184699908249, 0.0074887352748561585],
            lidar2ego_translation=[0.60465247, -0.00544473, 2.0398749],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = np.array(input_dict["lidar2ego_translation"])

        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(input_dict["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            for cam_type, cam_info in info["cams"].items():
                
                # 4. [DATA-DIFF] skip front zoom camera
                if cam_type == 'camera_fz':
                    continue

                # 5. [DATA-DIFF] use undistorted data path
                cam_info["img_path"] = cam_info["img_path"].replace('extracted', 'preproc')
                assert 'preproc' in cam_info['img_path'], f"{cam_info['img_path']}: old path"

                Tac = np.array(info['cams'][cam_type]['cam_extrinsic'])

                image_paths.append(cam_info["img_path"])
                # obtain lidar to image transformation matrix
                Tlc = np.linalg.inv(lidar2ego) @ Tac
                lidar2cam_rt = np.linalg.inv(Tlc)

                intrinsic = copy.deepcopy(INTRINSICS[cam_type])
                viewpad = np.eye(4)
                viewpad[0, 0] = intrinsic[0]
                viewpad[1, 1] = intrinsic[1]
                viewpad[0, 2] = intrinsic[2]
                viewpad[1, 2] = intrinsic[3]
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsic.append(viewpad)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_all_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_all_instance_inds = info.get('instance_inds', np.array([]))
        gt_all_velocity = info.get('gt_velocity', np.array([]))
        gt_all_velocity = np.nan_to_num(gt_all_velocity)


        gt_labels_3d = []
        gt_bboxes_3d = []
        gt_velocity = []
        gt_instance_ids = []

        # map for safead to nuscenes classes
        SAFEAD_CLASS_TO_NUSC = {
                            "Truck": "truck",
                            "Cyclist": "bicycle",
                            "Bus": "bus",
                            # "Vehicle_other": 200,
                            # "Moved by person": 200,
                            "Train": "traffic_cone",
                            "Car": "bicycle",
                            "Animal": "construction_vehicle",
                            "Pedestrian": "pedestrian",
                            "Motorcyclist": "motorcycle",
                            "Trailer": "trailer",
                            "Scooter": "construction_vehicle",
                        }

        for safead_cat, box, vel, inst_id in zip(gt_names_3d, gt_all_bboxes_3d, gt_all_velocity, gt_all_instance_inds):
            # [DATA-DIFF] map safead classes to nuscenes classes
            # if safead_cat in SAFEAD_CLASS_TO_NUSC:
            #     cat = SAFEAD_CLASS_TO_NUSC[safead_cat]
            # else:
            #     continue

            if safead_cat in list(SAFEAD_CLASS_TO_NUSC.values()):
                gt_labels_3d.append(self.CLASSES.index(safead_cat))
                gt_bboxes_3d.append(box)
                gt_velocity.append(vel)
                gt_instance_ids.append(inst_id)

            else:
                gt_labels_3d.append(-1)
                gt_bboxes_3d.append(box)
                gt_velocity.append(vel)
                gt_instance_ids.append(inst_id)


        gt_labels_3d = np.array(gt_labels_3d)

        gt_bboxes_3d = np.array(gt_bboxes_3d).reshape(-1, 7)
        gt_labels_3d = np.array(gt_labels_3d)
        gt_velocity = np.array(gt_velocity).reshape(-1, 2)
        gt_instance_ids = np.array(gt_instance_ids)

        if self.with_velocity:
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=gt_instance_ids,
        )

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = SafeAD3DDetTrackDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = SafeAD3DDetTrackDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        for metric in ["detection", "tracking"]:
            tracking = metric == "tracking"
            if tracking and not self.tracking:
                continue
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix, tracking=tracking
            )

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    ret_dict = self._evaluate_single(
                        result_files[name], tracking=tracking
                    )
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(
                    result_files, tracking=tracking
                )
            if tmp_dir is not None:
                tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        save_dir = "/home/goel/tmp/debug/sparse4d/visual"
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results[:1000]):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                else:
                    img = img.astype(np.uint8)
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):

                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        lidar2ego_rotation=[0.99972822181831, -0.016481156718805186, -0.014689184699908249, 0.0074887352748561585]
        lidar2ego_translation=[0.60465247, -0.00544473, 2.0398749]
        box.rotate(pyquaternion.Quaternion(lidar2ego_rotation))
        box.translate(np.array(lidar2ego_translation))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
