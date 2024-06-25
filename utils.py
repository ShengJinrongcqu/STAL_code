import os
# import matlab
import json
import subprocess
import numpy as np
import pandas as pd

from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
from scipy.interpolate import interp1d
from skimage import measure
from skimage.morphology import dilation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import medfilt
import torch
from torchvision.ops import nms
from Util.THcheck import TH14evalDet1 as detect_eval
import pdb

################ Config ##########################

# train.py --> 82
def load_config_file(config_file):
    '''
    -- Doc for parameters in the json file --

    feature_oversample:   Whether data augmentation is used (five crop and filp).是否数据增强
    sample_rate:          How many frames between adjacent feature snippet.相邻特征片段多少帧

    with_bg:              Whether hard negative mining is used.是否使用hard negative
    diversity_reg:        Whether diversity loss and norm regularization are used.是否使用diversity_loss和正则
    diversity_weight:     The weight of both diversity loss and norm regularization.diversity loss和正则的权重

    train_run_num:        How many times the experiment is repeated.训练次数
    training_max_len:     Crop the feature sequence when training if it exceeds this length.如果训练的特征序列的长度超过这个长度就裁剪它

    learning_rate_decay:  Whether to reduce the learning rate at half of training steps.是否在训练时减少学习率
    max_step_num:         Number of training steps.训练步数
    check_points:         Check points to test and save models.
    log_freq:             How many training steps the log is added to tensorboard.

    model_params:
    cls_branch_num:       Branch number in the multibranch network.分支网络中的分支号
    base_layer_params:    Filter number and size in each layer of the embedding module.在嵌入模块的每层中的过滤器数和大小
    cls_layer_params:     Filter number and size in each layer of the classification module.在分类模块的每层中的过滤器数和大小
    att_layer_params:     Filter number and size in each layer of the attention module.在attention模块的每层中的过滤器数和大小

    detect_params:        Parameters for action localization on the CAS. CAS上行为定位的参数
                          See detect.py for details.

    base_sample_rate:     'sample_rate' when feature extraction.特征提取时的'sample_rate'
    base_snippet_size:    The size of each feature snippet.每个特征片段的大小

    bg_mask_dir:          The folder of masks of static clips.静态剪辑掩码的文件夹

    < Others are easy to guess >

    '''

    all_params = json.load(open(config_file))

    dataset_name = all_params['dataset_name']

    # all_params['ac_paths'] = all_params['file_paths']['ActivityNet12']
    all_params['file_paths'] = all_params['file_paths'][dataset_name]

    all_params['action_class_num'] = all_params['action_class_num']
    all_params['base_sample_rate'] = all_params['base_sample_rate']
    all_params['base_snippet_size'] = all_params['base_snippet_size']
    all_params['train_subset_name'] = all_params['train_subset_name']

    assert (all_params['sample_rate'] % all_params['base_sample_rate'] == 0)

    all_params['model_class_num'] = all_params['action_class_num']
    all_params['model_params']['class_num'] = all_params['model_class_num']

    # Convert second to frames
    all_params['detect_params']['proc_value'] = int(
            all_params['detect_params']['proc_value'] * all_params['sample_rate'])

    # print(all_params)
    return all_params


################ Class Name Mapping #####################

thumos14_old_cls_names = {
    7: 'BaseballPitch',
    9: 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}

thumos14_old_cls_indices = {v: k for k, v in thumos14_old_cls_names.items()}

thumos14_new_cls_names = {
    0: 'BaseballPitch',
    1: 'BasketballDunk',
    2: 'Billiards',
    3: 'CleanAndJerk',
    4: 'CliffDiving',
    5: 'CricketBowling',
    6: 'CricketShot',
    7: 'Diving',
    8: 'FrisbeeCatch',
    9: 'GolfSwing',
    10: 'HammerThrow',
    11: 'HighJump',
    12: 'JavelinThrow',
    13: 'LongJump',
    14: 'PoleVault',
    15: 'Shotput',
    16: 'SoccerPenalty',
    17: 'TennisSwing',
    18: 'ThrowDiscus',
    19: 'VolleyballSpiking',
    20: 'Background',
}

thumos14_new_cls_indices = {v: k for k, v in thumos14_new_cls_names.items()}

old_cls_names = {
    'thumos14': thumos14_old_cls_names,
    'ActivityNet12': np.load('misc/old_cls_names_anet12.npy',allow_pickle=True).item(),
    'ActivityNet13': np.load('misc/old_cls_names_anet13.npy',allow_pickle=True).item(),
}

old_cls_indices = {
    'thumos14': thumos14_old_cls_indices,
    'ActivityNet12': np.load('misc/old_cls_indices_anet12.npy',allow_pickle=True).item(),
    'ActivityNet13': np.load('misc/old_cls_indices_anet13.npy',allow_pickle=True).item(),
}

new_cls_names = {
    'thumos14': thumos14_new_cls_names,
    'ActivityNet12': np.load('misc/new_cls_names_anet12.npy',allow_pickle=True).item(),
    'ActivityNet13': np.load('misc/new_cls_names_anet13.npy',allow_pickle=True).item(),
}
# utlis.py -->201
new_cls_indices = {
    'thumos14': thumos14_new_cls_indices,
    'ActivityNet12': np.load('misc/new_cls_indices_anet12.npy',allow_pickle=True).item(),
    'ActivityNet13': np.load('misc/new_cls_indices_anet13.npy',allow_pickle=True).item(),
}

################ Load dataset #####################

# utils.py --> 188
def load_meta(meta_file):
    '''Load video metas from the mat file (Only for thumos14).'''
    meta_data = loadmat(meta_file)
    meta_mat_name = [i for i in meta_data.keys() if 'videos' in i][0]  # 取结构名validation_videos
    meta_data = meta_data[meta_mat_name][0]
    return meta_data

# utils.py -->204
def load_annotation_file(anno_file):
    '''Load action instaces from a single file (Only for thumos14).'''
    anno_data = pd.read_csv(anno_file, header=None, delimiter=' ')  #对应着meta数据里的video_name, 如：video_validation
    anno_data = np.array(anno_data)
    return anno_data

# utils.py --> 468
def __get_thumos14_meta(meta_file, anno_dir):

    meta_data = load_meta(meta_file)  # 存着视频对应的数据

    dataset_dict = {}
    # 不要ambiguous和detclasslist
    anno_files = [i for i in os.listdir(anno_dir) if 'Ambiguous' not in i]
    anno_files.remove('detclasslist.txt')
    anno_files.sort()

    for anno_file in anno_files:

        action_label = anno_file.split('_')[0]  # 获取标签类别
        action_label = new_cls_indices['thumos14'][action_label] # 获取标签类别对应数字

        anno_file = os.path.join(anno_dir, anno_file)  # 拼出全路径
        anno_data = load_annotation_file(anno_file)  # 获取每个标签对应的
        # 分割出video_name start end
        for entry in anno_data:
            video_name = entry[0]
            start = entry[2]
            end = entry[3]

            ### Initializatiton ###  从meta_data 拿出对应数据，然后放入detaset_dict（是字典），一个video中可能会有多个时间
            if video_name not in dataset_dict.keys():
                # 找到meta_data的对应数据
                video_meta = [i for i in meta_data if i[0][0] == video_name][0]

                duration = video_meta[meta_data.dtype.names.index(
                    'video_duration_seconds')][0, 0]
                frame_rate = video_meta[meta_data.dtype.names.index(
                    'frame_rate_FPS')][0, 0]

                dataset_dict[video_name] = {
                    'duration': duration,
                    'frame_rate': frame_rate,
                    'labels': [],
                    'annotations': {},
                }

            if action_label not in dataset_dict[video_name]['labels']:
                dataset_dict[video_name]['labels'].append(action_label)
                dataset_dict[video_name]['annotations'][action_label] = []
            ###

            dataset_dict[video_name]['annotations'][action_label].append(
                [start, end])

    return dataset_dict


def __get_anet_meta(anno_json_file, dataset_name, subset):

    data = json.load(open(anno_json_file, 'r'))
    # taxonomy_data = data['taxonomy']
    database_data = data['database']
    missing_videos = np.load('misc/anet_missing_videos.npy')

    if subset == 'train':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'training'
        }
    elif subset == 'val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] == 'validation'
        }
    elif subset == 'train_and_val':
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v['subset'] in ['training', 'validation']
        }
    elif subset == 'test':
        subset_data = {
            k: v for k, v in database_data.items() if v['subset'] == 'testing'
        }

    dataset_dict = {}

    for video_name, v in subset_data.items():

        if video_name in missing_videos:
            print('Missing video: {}'.format(video_name))
            continue

        dataset_dict[video_name] = {
            'duration': v['duration'],
            'frame_rate': 25,  # ActivityNet should be formatted to 25 fps first
            'labels': [],
            'annotations': {},
        }

        for entry in v['annotations']:

            action_label = entry['label']
            action_label = new_cls_indices[dataset_name][action_label]

            if action_label not in dataset_dict[video_name]['labels']:
                dataset_dict[video_name]['labels'].append(action_label)
                dataset_dict[video_name]['annotations'][action_label] = []

            dataset_dict[video_name]['annotations'][action_label].append(
                entry['segment'])

    return dataset_dict


def __load_features(
        dataset_dict,  # dataset_dict will be modified
        dataset_name,
        sample_rate, # 总采样间隔数
        base_sample_rate,  # Untremed的采样间隔数
        temporal_aug,
        rgb_feature_dir,
        flow_feature_dir):


    f_sample_rate = int(sample_rate / base_sample_rate) # 本次采样

    # sample_rate of feature sequences, not original video

    ###############
    def __process_feature_file(filename):
        ''' Load features from a single file. '''

        feature_data = np.load(filename)

        frame_cnt = feature_data['frame_cnt'].item()  # 视频总帧数
        feature = feature_data['feature']

        # Feature: (B, T, F)
        # Example: (1, 249, 1024) or (10, 249, 1024) (Oversample)

        if temporal_aug:  # Data augmentation with temporal offsets
            feature = [
                feature[:, offset::f_sample_rate, :]
                for offset in range(f_sample_rate)
            ]
            # Cut to same length, OK when training
            min_len = int(min([i.shape[1] for i in feature]))
            feature = [i[:, :min_len, :] for i in feature]

            assert (len(set([i.shape[1] for i in feature])) == 1)
            feature = np.concatenate(feature, axis=0)

        else:
            feature = feature[:, ::f_sample_rate, :]

        return feature, frame_cnt

        # Feature: (B x f_sample_rate, T, F) T已经变成T/f_sample_rate,

    ###############

    # Load all features
    for k in dataset_dict.keys():

        print('Loading: {}'.format(k))

        # Init empty
        dataset_dict[k]['frame_cnt'] = -1
        dataset_dict[k]['rgb_feature'] = -1
        dataset_dict[k]['flow_feature'] = -1

        if rgb_feature_dir:

            if dataset_name == 'thumos14':  # 视频路径
                rgb_feature_file = os.path.join(rgb_feature_dir, k + '-rgb.npz')
            else:
                rgb_feature_file = os.path.join(rgb_feature_dir,
                                                'v_' + k + '-rgb.npz')

            rgb_feature, rgb_frame_cnt = __process_feature_file(
                rgb_feature_file)

            dataset_dict[k]['frame_cnt'] = rgb_frame_cnt
            dataset_dict[k]['rgb_feature'] = rgb_feature

        if flow_feature_dir:

            if dataset_name == 'thumos14':
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 k + '-flow.npz')
            else:
                flow_feature_file = os.path.join(flow_feature_dir,
                                                 'v_' + k + '-flow.npz')

            flow_feature, flow_frame_cnt = __process_feature_file(
                flow_feature_file)

            dataset_dict[k]['frame_cnt'] = flow_frame_cnt
            dataset_dict[k]['flow_feature'] = flow_feature

        if rgb_feature_dir and flow_feature_dir:
            assert (rgb_frame_cnt == flow_frame_cnt)
            assert (dataset_dict[k]['rgb_feature'].shape[1] == dataset_dict[k]
                    ['flow_feature'].shape[1])
            assert (dataset_dict[k]['rgb_feature'].mean() !=
                    dataset_dict[k]['flow_feature'].mean())

    return dataset_dict


def get_dataset(dataset_name,
                subset,
                file_paths,
                sample_rate,
                base_sample_rate,
                feature_oversample=True,
                temporal_aug=False):

    if dataset_name == 'thumos14':  # dataset_dict-->{video_name': {'duration': , 'frame_rate': , 'labels': , 'annotations': }}
        dataset_dict = __get_thumos14_meta(
            meta_file=file_paths[subset]['meta_file'],
            anno_dir=file_paths[subset]['anno_dir'])
    else:
        dataset_dict = __get_anet_meta(file_paths[subset]['anno_json_file'],
                                       dataset_name, subset)

    _temp_f_type = ('i3d' +
                    '-oversample' if feature_oversample else 'i3d' +
                    '-resize')
    # 取输入数据的地址

    rgb_dir = file_paths[subset]['feature_dir'][_temp_f_type]['rgb']
    flow_dir = file_paths[subset]['feature_dir'][_temp_f_type]['flow']

    dataset_dict = __load_features(dataset_dict, dataset_name,sample_rate, base_sample_rate, temporal_aug,
                                   rgb_dir, flow_dir)

    return dataset_dict


def get_single_label_dict(dataset_dict):
    '''
    If a video has multiple action classes, we treat it as multiple videos with
    single class. And the weight of each of them is reduced.
    '''
    new_dict = {}  # Create a new dict

    for k, v in dataset_dict.items():
        for label in v['labels']:

            new_key = '{}-{}'.format(k, label)

            new_dict[new_key] = dict(v)

            new_dict[new_key]['label_single'] = label
            new_dict[new_key]['annotations'] = v['annotations'][label]
            new_dict[new_key]['weight'] = (1 / len(v['labels']))

            new_dict[new_key]['old_key'] = k

    return new_dict  # This dict should be read only


def get_videos_each_class(dataset_dict):

    videos_each_class = defaultdict(list)

    for k, v in dataset_dict.items():

        if 'label_single' in v.keys():
            label = v['label_single']
            videos_each_class[label].append(k)

        else:
            for label in v['labels']:
                videos_each_class[label].append(k)

    return videos_each_class


def filter_segments(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        #s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

################ Post-Processing #####################


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def smooth(x):  # Two Dim nparray, On 1st dim
    temp = np.array(x)

    temp[1:, :] = temp[1:, :] + x[:-1, :]
    temp[:-1, :] = temp[:-1, :] + x[1:, :]

    temp[1:-1, :] /= 3
    temp[0, :] /= 2
    temp[-1, :] /= 2

    return temp


def __get_frame_ticks(frame_cnt, sample_rate, snippet_size=None):
    '''Get the frames of each feature snippet location.'''

    clipped_length = frame_cnt - snippet_size
    clipped_length = (clipped_length // sample_rate) * sample_rate
    # the start of the last chunk

    frame_ticks = np.arange(0, clipped_length + 1, sample_rate)
    # From 0, the start of chunks, clipped_length included

    return frame_ticks


def interpolate(x,
                frame_cnt,
                sample_rate,
                snippet_size=None,
                ):
    '''Upsample the sequence the original video fps.'''

    frame_ticks = __get_frame_ticks(frame_cnt, sample_rate,
                                    snippet_size)

    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1] + 1)
    # frame_ticks[-1] included

    interp_func = interp1d(frame_ticks, x, kind='linear')
    out = interp_func(full_ticks)

    return out


################ THUMOS Evaluation #####################


def eval_thumos_detect(detfilename, gtpath, subset, threshold):
    assert (subset in ['test', 'val'])

    # matlab_eng.addpath('THUMOS14_evalkit_20150930')
    # aps = matlab_eng.TH14evalDet(detfilename, gtpath, subset, threshold)
    aps, maps, cls_video_aps, cls_names = detect_eval(detfilename, gtpath, subset, threshold)
    # print(aps)
    aps = np.array(aps)
    mean_ap = aps.mean()
    # print(mean_ap)
    # print('The map: ', maps)
    # print(cls_video_aps)

    return aps, mean_ap, cls_video_aps, cls_names

################ Action Localization #####################


def detections_to_mask(length, detections):

    mask = np.zeros((length, 1))
    for entry in detections:
        mask[entry[0]:entry[1]] = 1

    return mask

# 根据掩码找到准确定位的帧
def mask_to_detections(mask, metric, weight_inner, weight_outter):

    out_detections = []
    detection_map = measure.label(mask, background=0)  # 实现连通区域标记
    detection_num = detection_map.max() # 多少个连通区域就是有多少动作

    for detection_id in range(1, detection_num + 1):

        start = np.where(detection_map == detection_id)[0].min()  # 定位起始和结束
        end = np.where(detection_map == detection_id)[0].max() + 1

        length = end - start

        inner_area = metric[detection_map == detection_id]

        left_start = min(int(start - length * 0.25),
                         start - 1)  # Context size 0.25
        right_end = max(int(end + length * 0.25), end + 1)
        # 为了算置信度
        outter_area_left = metric[left_start:start, :]
        outter_area_right = metric[end:right_end, :]

        outter_area = np.concatenate((outter_area_left, outter_area_right),
                                     axis=0)

        if outter_area.shape[0] == 0:
            detection_score = inner_area.mean() * weight_inner
        else:
            detection_score = (inner_area.mean() * weight_inner +
                               outter_area.mean() * weight_outter)

        out_detections.append([start, end, None, detection_score])

    return out_detections

def detect_with_thresholding(metric, thrh_value, proc_value, threshold):

    mask = metric > (thrh_value * threshold)
    out_detections = []


    # 因为通过阈值筛选，会导致动作不连续，通过膨胀连续
    mask = dilation(mask, np.array([[1] for _ in range(proc_value)]))

    return mask

# nms
def nms_pro(arr):
    p = [2, 0, 1]
    idx = np.argsort(p)
    numpy_arr = np.array(arr)
    prop_tensor = (numpy_arr[:, [0,1,3]])
    fake_y = np.tile(np.array([0, 1]), (numpy_arr.shape[0], 1))
    box = prop_tensor[:, :2]
    score = prop_tensor[:, 2]
    box_prop = np.concatenate((fake_y, box), 1)
    p2 = [0, 2, 1, 3]
    pidx = np.argsort(p2)
    box_prop = box_prop[:, pidx]
    box_prop = torch.Tensor(box_prop)
    score = torch.Tensor(score)
    result = nms(box_prop, score, 0.5)
    return result.numpy().astype(np.int)
################ Output Detection To Files ################


def output_detections_thumos14(out_detections, out_file_name):
    '''
    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names['thumos14'][class_id]
        old_class_id = int(old_cls_indices['thumos14'][class_name])
        entry[3] = old_class_id
    '''
    out_file = open(out_file_name, 'w')

    for entry in out_detections:
        out_file.write('{} {:.2f} {:.2f} {} {:.4f}\n'.format(
            entry[0], entry[1], entry[2], int(entry[3]), entry[4]))

    out_file.close()


def output_detections_anet(out_detections, out_file_name, dataset_name,
                           feature_type):

    assert (dataset_name in ['ActivityNet12', 'ActivityNet13'])
    assert (feature_type in ['untri', 'i3d'])

    for entry in out_detections:
        class_id = entry[3]
        class_name = new_cls_names[dataset_name][class_id]
        entry[3] = class_name

    output_dict = {}

    if dataset_name == 'ActivityNet12':
        output_dict['version'] = 'VERSION 1.2'
    else:
        output_dict['version'] = 'VERSION 1.3'

    if feature_type == 'untri':
        output_dict['external_data'] = {
            'used': False,
            'details': 'Untri feature'
        }
    else:
        output_dict['external_data'] = {'used': True, 'details': 'I3D feature'}

    output_dict['results'] = {}

    for entry in out_detections:

        if entry[0] not in output_dict['results']:
            output_dict['results'][entry[0]] = []

        output_dict['results'][entry[0]].append({
            'label': entry[3],
            'score': entry[4],
            'segment': [entry[1], entry[2]],
        })

    with open(out_file_name, 'w') as f:
        f.write(json.dumps(output_dict))


################ Visualization #####################


def get_snippet_gt(annos, fps, sample_rate, snippet_num):

    gt = np.zeros((snippet_num,))

    for i in annos:
        start = int(float(i[0]) * fps // sample_rate)
        end = int(float(i[1]) * fps // sample_rate)

        gt[start:start + 1] = 0.5
        gt[end:end + 1] = 0.5
        gt[start + 1:end] = 1

    return gt


def visualize_scores_barcodes(score_titles,
                              scores,
                              ylim=None,
                              out_file=None,
                              show=False):

    lens = [i.shape[0] for i in scores]
    assert (len(set(lens)) == 1)
    frame_cnt = lens[0]  # Not all frame are visualized, clipped at end

    subplot_sum = len(score_titles)

    fig = plt.figure(figsize=(20, 10))

    height_ratios = [1 for _ in range(subplot_sum)]

    gs = gridspec.GridSpec(subplot_sum, 1, height_ratios=height_ratios)

    for j in range(len(score_titles)):

        fig.add_subplot(gs[j])

        plt.xticks([])
        plt.yticks([])

        plt.title(score_titles[j], position=(-0.1, 0))

        axes = plt.gca()

        if j == 0:
            barprops = dict(aspect='auto',
                            cmap=plt.cm.PiYG,
                            interpolation='nearest',
                            vmin=-1,
                            vmax=1)
        elif j == 1:
            barprops = dict(aspect='auto',
                            cmap=plt.cm.seismic,
                            interpolation='nearest',
                            vmin=-1,
                            vmax=1)
        elif j == 2 or j == 3:

            if ylim:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Purples,
                                interpolation='nearest',
                                vmin=ylim[0],
                                vmax=ylim[1])
            else:
                barprops = dict(
                    aspect='auto',
                    cmap=plt.cm.Purples,  #BrBG
                    interpolation='nearest')

        else:
            if ylim:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Blues,
                                interpolation='nearest',
                                vmin=ylim[0],
                                vmax=ylim[1])
            else:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.Blues,
                                interpolation='nearest')

        axes.imshow(scores[j].reshape((1, -1)), **barprops)

    if out_file:
        plt.savefig(out_file)

    if show:
        plt.show()

    plt.close()


def visualize_video_with_scores_barcodes(images_dir,
                                         images_prefix,
                                         score_titles,
                                         scores,
                                         out_file,
                                         fps,
                                         ylim=None):  # Fps: original video fps

    images_paths = [
        os.path.join(images_dir, i)
        for i in os.listdir(images_dir)
        if i.startswith(images_prefix)
    ]

    images_paths.sort()

    lens = [i.shape[0] for i in scores]
    assert (len(set(lens)) == 1)
    frame_cnt = lens[0]  # Not all frame are visualized, clipped at end

    subplot_sum = len(score_titles) + 1

    temp_dir = './temp_plots'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for i in range(frame_cnt):

        fig = plt.figure(figsize=(15, 10))

        height_ratios = [1 for _ in range(subplot_sum)]
        height_ratios[0] = 12

        gs = gridspec.GridSpec(subplot_sum, 1, height_ratios=height_ratios)

        fig.add_subplot(gs[0])

        plt.axis('off')
        plt.title('Video')
        plt.imshow(Image.open(images_paths[i]).convert('RGB'))

        for j in range(len(score_titles)):

            fig.add_subplot(gs[j + 1])

            plt.xticks([])
            plt.yticks([])

            plt.title(score_titles[j], position=(-0.1, 0))

            axes = plt.gca()

            if j == 0:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.PiYG,
                                interpolation='nearest',
                                vmin=-1,
                                vmax=1)
            elif j == 1:
                barprops = dict(aspect='auto',
                                cmap=plt.cm.seismic,
                                interpolation='nearest',
                                vmin=-1,
                                vmax=1)
            elif j == 2 or j == 3:

                if ylim:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Purples,
                                    interpolation='nearest',
                                    vmin=ylim[0],
                                    vmax=ylim[1])
                else:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Purples,
                                    interpolation='nearest')

            else:
                if ylim:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Blues,
                                    interpolation='nearest',
                                    vmin=ylim[0],
                                    vmax=ylim[1])
                else:
                    barprops = dict(aspect='auto',
                                    cmap=plt.cm.Blues,
                                    interpolation='nearest')

            axes.imshow(scores[j].reshape((1, -1)), **barprops)

            axes.axvline(x=i, color='darkorange')

        plt.savefig(os.path.join(temp_dir, '{:0>6}.png'.format(i)))
        plt.close()

    subprocess.call([
        'ffmpeg', '-framerate',
        str(fps), '-i',
        os.path.join(temp_dir, '%06d.png'), '-pix_fmt', 'yuv420p', out_file
    ])

    os.system('rm -r {}'.format(temp_dir))
