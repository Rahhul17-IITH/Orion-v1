data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="B2DOrionDataset",
        data_root="data/bench2drive",
        ann_file="data/infos/b2d_infos_train.pkl",
        map_file="data/infos/b2d_map_infos.pkl",
        pipeline=[PhotoMetricDistortionMultiViewImage(), NormalizeMultiviewImage([123.675, 116.28, 103.53],[58.395,57.12,57.375]), PadMultiViewImage(32),
                  PETRFormatBundle3D(class_names=['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'], collect_keys=['img','ego_fut_trajs','input_ids','gt_labels_3d']),
                  CustomCollect3D(['img','ego_fut_trajs','input_ids','gt_labels_3d'])
                  ],
        classes=['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'],
        modality=dict(use_camera=True),
        queue_length=1,
        past_frames=2,
        future_frames=6,
    )
)

model = dict(
    loss_plan_reg=dict(loss_weight=3.0),
    loss_vae_gen=dict(loss_weight=3.0)
)

optimizer = dict(
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
