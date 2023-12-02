from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config
from mmengine.runner import Runner


# define dataset root and directory for images and annotations
data_root = 'idd20kII'
train_img_dir = 'train_img'
train_ann_dir = 'train_label'
val_img_dir = 'val_img'
val_ann_dir = 'val_label'

# define class and palette for better visualization
classes = ('road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'person', 'rider', 'motorcycle',
            'bicycle', 'auto-rickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence',
              'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs fallback', 'building', 'bridge',
                'vegetation', 'sky', 'unlabelled')
palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [255]]


@DATASETS.register_module()
class IDD_Data(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

#edit cfg
cfg = Config.fromfile('deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.crop_size = (512,512)
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 27
cfg.model.auxiliary_head.num_classes = 27

# Modify dataset type and path
cfg.dataset_type = 'IDD_Data'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 4

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1080,1920), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1080, 1920), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=train_img_dir, seg_map_path=train_ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=val_img_dir, seg_map_path=val_ann_dir)
#here change was for val to train dir!
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader


# Load the pretrained weights
# cfg.load_from = 'deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/tutorial4'

cfg.train_cfg.max_iters = 80000
cfg.train_cfg.val_interval = 1000
cfg.default_hooks.logger.interval = 500
cfg.default_hooks.checkpoint.interval = 10000

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
print("Final Config\n\n\n\n\n")
print(f'Config:\n{cfg.pretty_text}')

runner = Runner.from_cfg(cfg)
runner.train()
