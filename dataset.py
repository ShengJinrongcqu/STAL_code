
import random
import numpy as np
from torch.utils.data import Dataset

from utils import get_single_label_dict
import pdb

# dataset.py 随机选择数据增强序列
def _random_select(rgb=-1, flow=-1):
    ''' Randomly select one augmented feature sequence. '''

    if type(rgb) != int and type(flow) != int:

        assert (rgb.shape[0] == flow.shape[0])
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])
        flow = np.array(flow[random_idx, :, :])

    elif type(rgb) != int:
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])

    elif type(flow) != int:
        random_idx = random.randint(0, flow.shape[0] - 1)
        flow = np.array(flow[random_idx, :, :])
    else:
        pass

    return rgb, flow

def _check_length(rgb, flow, max_len):

    if type(rgb) != int and type(flow) != int:

        assert (rgb.shape[1] == flow.shape[1])
        if rgb.shape[1] > max_len:
            print('Crop Both!')
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start:start + max_len, :])
            flow = np.array(flow[:, start:start + max_len, :])

    elif type(rgb) != int:

        if rgb.shape[1] > max_len:
            print('Crop RGB!')
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start:start + max_len, :])

    elif type(flow) != int:

        if flow.shape[1] > max_len:
            print('Crop FLOW!')
            start = random.randint(0, flow.shape[1] - max_len)
            flow = np.array(flow[:, start:start + max_len, :])
    else:
        pass

    return rgb, flow

class SingleVideoDataset(Dataset):

    def __init__(self,
                 dataset_dict,
                 random_select=False):

        self.dataset_dict = dataset_dict
        self.random_select = random_select
        self.video_list = list(self.dataset_dict.keys())
        self.video_list.sort()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video = self.video_list[idx]
        # 根据索引获取相关数据
        rgb, flow = (self.dataset_dict[video]['rgb_feature'],
                     self.dataset_dict[video]['flow_feature'])

        # 输入的一个视频的特征机上数据增强有n种，随机选择用于训练
        if self.random_select:
            rgb, flow = _random_select(rgb, flow)

        return_dict = {
            'video_name': video,
            'rgb': rgb,
            'flow': flow,
            'frame_rate': self.dataset_dict[video]['frame_rate'],  # frame_rate == fps
            'frame_cnt': self.dataset_dict[video]['frame_cnt'],
            'anno': self.dataset_dict[video]['annotations'],
            'label': self.dataset_dict[video]['labels']
        }
        return return_dict
