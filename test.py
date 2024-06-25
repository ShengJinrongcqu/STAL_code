import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from model  import SelfNetwork
from utils import eval_thumos_detect, detect_with_thresholding
from utils import get_dataset, normalize, interpolate, nms_pro
from utils import mask_to_detections, load_config_file
from utils import output_detections_thumos14, output_detections_anet
from dataset import SingleVideoDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda')

def softmax(x, dim):
    x = F.softmax(torch.from_numpy(x), dim=dim)
    return x.numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str)
    args = parser.parse_args()
    all_params = load_config_file(args.config_file)
    locals().update(all_params)

    def detect(data, dataset_dict):
        out_detections = []
        for video_name in data.keys():
            avg_score = data[video_name]['cas']
            att_weight = data[video_name]['att']  # attention流的权重
            global_score = data[video_name]['class_result'].squeeze(0)
            exsiting = data[video_name]['global_score']  # 一维

            duration = dataset_dict[video_name]['duration']
            fps = dataset_dict[video_name]['frame_rate']
            frame_cnt = dataset_dict[video_name]['frame_cnt']

            for class_id in exsiting:
                total_out = []
                temp = []

                _score = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                metric = (att_weight + (_score)) / 2

                if (len(metric) == 1):
                    break
                else:
                    metric = normalize(metric)
                metric = interpolate(metric[:, 0],
                                     frame_cnt,
                                     sample_rate,
                                     snippet_size=base_snippet_size)

                metric = np.expand_dims(metric, axis=1)  # 加维度

                m = detect_params['multi_scale']
                for i in range(-m, m + 1):
                    threshoding = metric.mean() + 0.025 * i
                    if threshoding <= 0 or threshoding >= 1:
                        continue

                    mask = detect_with_thresholding(metric, detect_params['thrh_value'], detect_params['proc_value'],
                                                    threshoding)

                    temp_out = mask_to_detections(mask, metric, detect_params['weight_inner'],
                                                  detect_params['weight_outter'])
                    for entry in temp_out:
                        entry[2] = class_id
                        entry[3] += global_score[class_id] * detect_params['weight_global']
                        total_out.append(entry)

                #########################################
                # non-maximum suppression
                if len(total_out) > 0:
                    nms_idx = nms_pro(total_out)
                    for k in nms_idx:
                        temp.append(total_out[k])
                for entry in temp:
                    entry[0] = (entry[0] + detect_params['sample_offset']) / fps
                    entry[1] = (entry[1] + detect_params['sample_offset']) / fps
                    entry[0] = max(0, entry[0])
                    entry[1] = max(0, entry[1])
                    entry[0] = min(duration, entry[0])
                    entry[1] = min(duration, entry[1])

                for entry_id in range(len(temp)):
                    temp[entry_id] = [video_name] + temp[entry_id]

                out_detections += temp




        return out_detections

    def get_features(loader, model):
        prediction = {}
        class_threshold = detect_params["global_score_thrh"]

        model.eval()



        for _, data in enumerate(loader):

            video_name = data['video_name'][0]

            rgb = data['rgb'].to(device).squeeze(0)  # 1 at dim0
            flow = data['flow'].to(device).squeeze(0)
            cat = torch.cat([rgb, flow], dim=2)
            with torch.no_grad():
                out,_ = model.forward(cat.transpose(2, 1))
                avg_score = out['cas']
                attention = out['att']
                class_result = out['clf']

                attention = attention[:, :, 0:1].cpu().numpy().mean(0)
                class_result = F.softmax(class_result, dim=1)[:, 0: action_class_num - 1]
                class_result = class_result.cpu().numpy().mean(0, keepdims=True)
                avg_score = avg_score.mean(0).cpu().numpy()

                if class_result.max() > class_threshold:
                    global_score = np.where(class_result > class_threshold)[1]
                else:
                    global_score = np.array(np.argmax(class_result), dtype=np.int).reshape(-1)

            data_dict = {
                "cas" : avg_score,
                "att" : attention,
                "class_result" : class_result,
                "global_score" : global_score
            }
            prediction[video_name] = data_dict

        return prediction

    test_dataset_dict = get_dataset(
        dataset_name=dataset_name,
        subset=test_subset_name,
        file_paths=file_paths,
        sample_rate=sample_rate,
        base_sample_rate=base_sample_rate,
        feature_oversample=feature_oversample,
        temporal_aug=False,
    )


    test_detect_dataset = SingleVideoDataset(test_dataset_dict,
                                             random_select=False)

    test_detect_loader = torch.utils.data.DataLoader(test_detect_dataset,
                                                     batch_size=1,
                                                     pin_memory=True,
                                                     shuffle=False)

    best_map = []
    best_maps = 0.0
    file = open('{}'.format(outfile_path), 'w')

    for cp_idx, check_point in enumerate(check_points):

        model = SelfNetwork(in_features=feature_dim * 2,
                                **model_params).to(device)
        model.load_state_dict(
            torch.load(
                os.path.join('models', experiment_naming,
                                'model-{}-2.0'.format(check_point))))

        ####################  获取预测行为  ##################
        print('model-{}---------------'.format(check_point))
        file.write('#####################################################\n')
        file.write('model-{}---------------------------------------------\n'.format(check_point))
        prediction = get_features(test_detect_loader, model)
        out_detections = detect(prediction, test_dataset_dict)

        pred_dir = os.path.join('outputs', 'predictions')
        test_pred_file = os.path.join(
            pred_dir,
            '{}-run-{}-test'.format(experiment_naming, check_point))

        if dataset_name == 'thumos14':
            output_detections_thumos14(out_detections, test_pred_file )
        elif dataset_name in ['ActivityNet12', 'ActivityNet13']:
            output_detections_anet(out_detections, test_pred_file, dataset_name)

        ####################  获取预测结果，ActivityNet数据需要额外提交服务器  ##################

        if dataset_name == 'thumos14':
            temp_resualts = []
            cur_all = 0.0
            cur_content = "cur:  "
            max_content = "max:  "
            for IoU_idx, IoU in enumerate(
                    [.1, .2, .3, .4, .5, .6, .7]):
                if len(out_detections) != 0:
                    temp_aps, temp_map, _, _ = eval_thumos_detect(test_pred_file,
                        file_paths[test_subset_name]['anno_dir'],
                        test_subset_name, IoU)
                    temp_resualts.append(temp_map * 10000 // 10)
                    cur_all += (temp_map * 10000 // 10)
                else:
                    print('Empty Detections')
            if cur_all / 7 > best_maps:
                best_maps = cur_all / 7
                best_map = temp_resualts.copy()
            for i in range(len(temp_resualts)):
                cur_content += ("IoU {}".format((i+1)/10) + "：%.3f   " % (temp_resualts[i] / 10))
                max_content += ("IoU {}".format((i+1)/10) + "：%.3f   " % (best_map[i] / 10))
            cur_content += "Avg IoU：%.3f" % (cur_all / 70)
            max_content += "Avg IoU：%.3f" % (best_maps / 10)
            file.write(cur_content + "\n")
            file.write(max_content + "\n")
            print(cur_content)
            print(max_content)


    file.close()




