model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='smpl24', mode='stgcn_spatial'),
        num_person=1),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=256))

dataset_type = 'PoseDataset'
#ann_file = 'data/aist++/aist++_smpl24_60.pkl' # for training
ann_file = 'data/aist++/aist++_smpl24_420.pkl' # for testing
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl24', feats=['jm']),
    dict(type='UniformSample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl24', feats=['jm']),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl24', feats=['jm']),
    dict(type='ContinuousSample', clip_len=60, num_clips=7),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
        val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
        test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0001, by_epoch=False)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs_4C_2/stgcn/stgcn_aist++_smpl24_420/jm'