from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train


model.backbone.bottom_up.freeze_at = 2
model.num_classes = 20

train.init_checkpoint = "/opt/workdir/_fcos_coco_orig_split/output_1989088_fcos_R_50_FPN_1x_5/model_final.pth"
train.max_iter = 18000
train.eval_period = 8000
train.seed = 5

dataloader.train.mapper.use_instance_mask = False

for augmentation in dataloader.train.mapper.augmentations:
    if hasattr(augmentation, 'short_edge_length'):
        augmentation.short_edge_length = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


for augmentation in dataloader.test.mapper.augmentations:
    if hasattr(augmentation, 'short_edge_length'):
        augmentation = 800

dataloader.train.dataset.names = 'voc_train_5'
dataloader.test.dataset.names = 'voc_val_5'

lr_multiplier.scheduler.milestones = [12000, 16000, 18000]
lr_multiplier.scheduler.values = [1, 0.1, 0.01]
lr_multiplier.warmup_length = 0.55
