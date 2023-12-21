from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.optim import SGD as optimizer
from ..common.train import train

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 8

train.init_checkpoint = "/opt/workdir/_fcos_coco_orig_split/output_1989088_fcos_R_50_FPN_1x_5/model_final.pth"
train.max_iter = 24000
train.eval_period = 8000
train.seed = 21

dataloader.train.mapper.use_instance_mask = False

for augmentation in dataloader.train.mapper.augmentations:
    if hasattr(augmentation, 'short_edge_length'):
        augmentation.short_edge_length = [800, 832, 864, 896, 928, 960, 992, 1024]
        augmentation.max_size = 2048
        augmentation.sample_style = 'choice'
        break

for augmentation in dataloader.test.mapper.augmentations:
    if hasattr(augmentation, 'short_edge_length'):
        augmentation.short_edge_length = 1024
        augmentation.max_size = 2048
        break


dataloader.train.dataset.names = 'cityscapes_train_21'
dataloader.test.dataset.names = 'cityscapes_val_21'
dataloader.total_batch_size = 8

lr_multiplier.scheduler.milestones = [18000, 24000]
lr_multiplier.scheduler.values = [1, 0.1]
optimizer.lr = 0.01
