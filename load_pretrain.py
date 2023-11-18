# from mmaction.apis import init_recognizer, inference_recognizer


# config_file = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint_file = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

# model = init_recognizer(config_file, checkpoint_file, device='cpu')

import os
import torch
from torchview import draw_graph
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')

# Test one video
video_dir = '/scratch-shared/tlong01/dataset/Moments_in_Time_256x256_30fps/training/eating'
video_files = os.listdir(video_dir)
vidoe_path = os.path.join(video_dir, video_files[0])

# [batch_size, channel, temporal_dim, height, width]
dummy_x = torch.rand(2, 3, 32, 224, 224)

# SwinTransformer3D without cls_head
backbone = model.backbone

with torch.no_grad():
    # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
    feat = backbone(dummy_x)
    feat1d = model.cls_head.avg_pool(feat).squeeze()

# alternative way
# feat = model.extract_feat(dummy_x)

# visualize the network
# model_graph = draw_graph(model, input_data=[dummy_x, torch.zeros(2,1)], save_graph = True, filename = 'model_graph')

