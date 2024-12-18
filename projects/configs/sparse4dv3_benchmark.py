# ================ base config ===================
_base_ = ['./default_runtime.py']

custom_imports = dict(imports=['projects.mmdet3d_plugin'], allow_failed_imports=False)

work_dir = None

batch_size_per_gpu = 8
num_gpus = 1
num_epochs = 10

compile = False
input_shape = (1408, 512)

tracking_test = True
tracking_threshold = 0.2

# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

metainfo = dict(classes=class_names)

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True

anno_root = "./data/nuscenes_anno_pkls/"

model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]
    ),
    depth_branch=None,
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="nusc_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
dataset_type = "CustomNuScenes3DDetTrackDataset"
data_root = "data/nuscenes"

train_pipeline = [
    dict(type="CustomLoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="BaseLoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type='CustomPack3DDetInputs',
        keys=["img", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "T_global", "T_global_inv", "timestamp", "timestamp", "projection_mat", "image_wh", "lidar2img"
        ],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
    with_seq_flag=True,
    keep_consistent_seq_aug=True
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
    "rot3d_range": [-0.3925, 0.3925],
}

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='GroupInBatchSampler'),
    dataset=dict(
        **data_basic_config,
        ann_file=anno_root + "/nuscenes-mini_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        metainfo=metainfo,
    )
)

val_dataloader = None
est_dataloader = None

# ================== training ========================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, weight_decay=0.001),
    paramwise_cfg=dict(custom_keys={
        "img_backbone": dict(lr_mult=0.5),
    }),
    clip_grad=dict(max_norm=25, norm_type=2),
    loss_scale=32.0,
)

# train params
total_batch_size = batch_size_per_gpu * num_gpus
num_iters_per_epoch = int(28130 // total_batch_size)
eval_interval = 20 * num_iters_per_epoch

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=500,
        end=num_iters_per_epoch * num_epochs,
        T_max=num_iters_per_epoch * num_epochs - 500,
        eta_min_ratio=1e-3,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=num_iters_per_epoch * num_epochs, val_interval=eval_interval
)
val_cfg = None
test_cfg = None

val_evaluator = None
test_evaluator = None

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=eval_interval, max_keep_ckpts=10, by_epoch=False))
visualizer = None

