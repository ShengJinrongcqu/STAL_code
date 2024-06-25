import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import smooth
from utils import eval_thumos_detect, detect_with_thresholding
from utils import get_dataset, normalize, interpolate, nms_pro
from utils import mask_to_detections, load_config_file
from utils import output_detections_thumos14, output_detections_anet

import pdb


def softmax(x, dim):
    x = F.softmax(torch.from_numpy(x), dim=dim)
    return x.numpy()

def gaussian(cas):

    x2 = np.zeros(cas.shape)
    x1 = np.ones(cas.shape)
    b = np.where(cas>0.03, x1, x2)
    return b


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config-file', type=str)
    parser.add_argument('--train-subset-name', type=str)
    parser.add_argument('--test-subset-name', type=str)

    parser.add_argument('--include-train',
                        dest='include_train',
                        action='store_true')
    parser.add_argument('--no-include-train',
                        dest='include_train',
                        action='store_false')
    parser.set_defaults(include_train=True)

    args = parser.parse_args()

    print(args.config_file)
    print(args.train_subset_name)
    print(args.test_subset_name)
    print(args.include_train)

    all_params = load_config_file(args.config_file)
    locals().update(all_params)


    train_dataset_dict = None

    test_dataset_dict = get_dataset(
        dataset_name=dataset_name,
        subset=args.test_subset_name,
        file_paths=file_paths,
        sample_rate=sample_rate,
        base_sample_rate=base_sample_rate,
        feature_type=feature_type,
        feature_oversample=False,
        temporal_aug=False,
    )

    dataset_dicts = {'train': train_dataset_dict, 'test': test_dataset_dict}

    def detect(
            cas_dir,
            subset,
            out_file_name,
            global_score_thrh,
            proc_value,
            sample_offset
    ):



        out_detections = []

        dataset_dict = dataset_dicts[subset]

        for video_name in dataset_dict.keys():
            # 获取处理好的数据的路径名，加载数据
            cas_file = video_name + '.npz'
            cas_data = np.load(os.path.join(cas_dir, cas_file))
            # avg_score"是Tx21(类别)
            avg_score = cas_data['avg_score']
            att_weight = cas_data['weight']  # attention流的权重

            # avg_score = cas_data['latent_avg_score']
            # att_weight = cas_data['latent_weight']

            global_score = cas_data['class_result'].squeeze(0)
            exsiting = cas_data['global_score']

            duration = dataset_dict[video_name]['duration']
            fps = dataset_dict[video_name]['frame_rate']
            frame_cnt = dataset_dict[video_name]['frame_cnt']


            ################ Threshoding ################
            for class_id in exsiting:
                total_out = []
                temp = []

                # 类别分数小于阈值则当不存在该类别


                _score = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                metric = (att_weight * 0.5 + _score * 0.5) #  * gaussian(_score) #* att_weight

                # metric = smooth(metric)
                if (len(metric) == 1):
                    break
                else:
                    metric = normalize(metric)
                # metric[metric<0.1]=0



                #########################################
                # 用插值的方式还原整个视频的分数

                metric = interpolate(metric[:, 0],
                                     frame_cnt,
                                     sample_rate,
                                     snippet_size=base_snippet_size,
                                     )

                metric = np.expand_dims(metric, axis=1) # 加维度


                m = 12

                for i in range(-m, m+1):
                    threshoding =  metric.mean() + 0.025 * i
                    if threshoding <= 0 or threshoding >=1:
                        continue
                    mask = detect_with_thresholding(metric, proc_value,threshoding)

                    temp_out = mask_to_detections(mask, metric, 1,
                                            -1)
                    for entry in temp_out:
                        entry[2] = class_id
                        entry[3] += global_score[class_id] * 0.25
                        total_out.append(entry)


                #########################################


                # non-maximum suppression


                if len(total_out) > 0:
                    nms_idx = nms_pro(total_out)
                    for k in nms_idx:
                        temp.append(total_out[k])
                for entry in temp:

                    entry[0] = (entry[0] + sample_offset) / fps
                    entry[1] = (entry[1] + sample_offset) / fps

                    entry[0] = max(0, entry[0])
                    entry[1] = max(0, entry[1])
                    entry[0] = min(duration, entry[0])
                    entry[1] = min(duration, entry[1])

                for entry_id in range(len(temp)):
                    temp[entry_id] = [video_name] + temp[entry_id]

                out_detections += temp


        if dataset_name == 'thumos14':
            output_detections_thumos14(out_detections, out_file_name)
        elif dataset_name in ['ActivityNet12', 'ActivityNet13']:
            output_detections_anet(out_detections, out_file_name, dataset_name,
                                   feature_type)

        return out_detections

    if dataset_name == 'thumos14':
        # summary_file = './outputs/summary-{}.npz'.format(experiment_naming)
        summary_file = './outputs/summary_trained-{}.npz'.format(experiment_naming)

        all_train_map = np.zeros((1, len(check_points), 4, 9, 1))
        all_train_aps = np.zeros(
            (1, len(check_points), 4, 9, action_class_num-1))
        all_test_map = np.zeros((1, len(check_points), 4, 9, 1))  # （3,1,4,9,1）
        all_test_aps = np.zeros(
            (1, len(check_points), 4, 9, action_class_num-1))


    for cp_idx, check_point in enumerate(check_points):

        for mod_idx, modality in enumerate(
            ['both']):
                # test.py已处理好的数据

            cas_dir = os.path.join(
                    'cas-features',
                    '{}-run-{}-{}'.format(experiment_naming,
                    check_point, modality))


            # cas_dir = "/data/codes/watl_ver0/Output/cas/thumos-I3D_motion_dif_m_s_i_hf_2021_step18000_new2_0_nm_shift3_d2_02-run-0-5000-both-test"
            # cas_dir = "/data/codes/watl_ver0/Output/cas/thumos-I3D_basenet3_1_s_pure_step8000-run-0-8000-late-fusion-test"
            # cas_dir = "/data/codes/watl_ver0/Output/cas/thumos-I3D_base_cas_tk_2021_step8000_new2_0_m-run-0-6000-both-test"
            # 输出
            pred_dir = os.path.join('outputs', 'predictions')

            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)


            test_pred_file = os.path.join(
                pred_dir,
                '{}-run-{}-{}-test'.format(experiment_naming,
                                            check_point, modality))


            print(mod_idx)
            test_outs = detect(cas_dir,'test', test_pred_file,
                                 **detect_params)

            if dataset_name == 'thumos14':

                for IoU_idx, IoU in enumerate(
                        [.1, .2, .3, .4, .5, .6, .7, .8, .9]):

                    if args.include_train:
                        if len(train_outs) != 0:
                            temp_aps, temp_map,_,_ = eval_thumos_detect(
                                    train_pred_file, file_paths[
                                    args.train_subset_name]['anno_dir'],
                                    args.train_subset_name, IoU)

                            all_train_aps[run_idx, cp_idx, mod_idx,
                                            IoU_idx, :] = temp_aps
                            all_train_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                          0] = temp_map
                        else:
                            print('Empty Detections')
                            all_train_aps[run_idx, cp_idx, mod_idx,
                                              IoU_idx, :] = 0
                            all_train_map[run_idx, cp_idx, mod_idx, IoU_idx,
                                              0] = 0

                    if len(test_outs) != 0:
                        temp_aps, temp_map,_,_ = eval_thumos_detect(
                            test_pred_file,
                            file_paths[args.test_subset_name]['anno_dir'],
                            args.test_subset_name, IoU)

                        all_test_aps[0, cp_idx, mod_idx,
                                      IoU_idx, :] = temp_aps
                        all_test_map[0, cp_idx, mod_idx, IoU_idx,
                                         0] = temp_map
                    else:
                        print('Empty Detections')
                        all_test_aps[0, cp_idx, mod_idx,
                                         IoU_idx, :] = 0
                        all_test_map[0, cp_idx, mod_idx, IoU_idx,
                                         0] = 0

                    print('{}{}{}'.format(cp_idx, mod_idx,
                                                IoU_idx))

    if dataset_name == 'thumos14':

        np.savez(summary_file,
                 all_train_aps=all_train_aps,
                 all_train_map=all_train_map,
                 all_test_aps=all_test_aps,
                 all_test_map=all_test_map)
