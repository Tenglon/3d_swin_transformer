import os
import torch
from pathlib import Path
from tqdm import tqdm
from torchview import draw_graph
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.apis import inference_recognizer


config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
model = model.cuda()
model.cfg = cfg
load_checkpoint(model, checkpoint, map_location='cpu')

# test a single video and show the result:
# video = 'demo.mp4'
# labelset_file = 'label_map_k400.txt'
labelset_file = '/scratch-shared/tlong01/dataset/Moments_in_Time_256x256_30fps/moments_categories.txt'

# Test one video
video_dir = '/scratch-shared/tlong01/dataset/Moments_in_Time_256x256_30fps/training/eating'
video_files = os.listdir(video_dir)

feat_dir = '/scratch-shared/tlong01/dataset/mim_3dvit_feat/train/eating'
os.makedirs(feat_dir, exist_ok=True)

for video_path in tqdm(video_files):
    video_path = os.path.join(video_dir, video_path)
    output_path = os.path.join(feat_dir, Path(video_path).stem + '.pth')
    if os.path.exists(output_path):
        print('Already exists {}'.format(output_path))
        continue
    results, features = inference_recognizer(model, video_path, labelset_file, outputs=['cls_head.avg_pool'])
    feat = features['cls_head.avg_pool'].squeeze()

    torch.save(feat, output_path)
    print('Saved to {}'.format(output_path))

# test load feature
feat_path = '/scratch-shared/tlong01/dataset/mim_3dvit_feat/train/eating/flickr-7-0-9-9-9-9-8-6-22470999986_1.pth'
feat = torch.load(feat_path)
print(feat.shape)
