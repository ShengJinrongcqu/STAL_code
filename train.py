from __future__ import absolute_import
import torch
import numpy as np
import argparse
import torch.optim as optim
import os
import random
from model import  SelfNetwork
from dataset import SingleVideoDataset
from utils import get_dataset, load_config_file
from loss import Losses
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(train_loader, naming):

    save_dir = os.path.join('models', naming)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = SelfNetwork(in_features=feature_dim * 2, **model_params).to(device)
    model_loss = Losses(class_num=action_class_num, loss_bg_weight=bg_weight,
                        loss_sim_weight=similarity_weight).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                            weight_decay=weight_decay)
    if learning_rate_decay:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[6000, 12000, 18000], gamma=0.1)

    optimizer.zero_grad()

    update_step_idx = 0
    single_video_idx = 0
    loss_recorder = {
        'cls': 0,
        'bg': 0,
        'sim': 0,
        'sum': 0,
    }

    while update_step_idx < max_step_num:
        # Train loop
        for _, data in enumerate(train_loader):

            model.train()
            single_video_idx += 1

            labels = data['label']
            rgb = data['rgb'].cuda()
            flow = data['flow'].cuda()
            model_input = torch.cat([rgb, flow], dim=2)

            out_ori, out_aug = model(model_input.transpose(2, 1))
            # out_ori = model(model_input.transpose(2, 1))

            loss, recoder = model_loss(out_ori, out_aug, labels)
            # loss, recoder = model_loss(out_ori, labels)
            loss_recorder['cls'] += recoder['clf']
            loss_recorder['bg'] += recoder['bg']
            loss_recorder['sim'] += recoder['sim']
            loss_recorder['sum'] += loss.item()
            loss.backward()

            # Update
            if single_video_idx % batch_size == 0:

                update_step_idx += 1
                print('Step {}: '.format(update_step_idx))
                print('Loss_clf-%.6f     Loss_bg-%.6f     Loss_sim-%.6f     Loss_sum-%.6f'%(
                    loss_recorder['cls'] / batch_size, loss_recorder['bg'] / batch_size,
                    loss_recorder['sim'] / batch_size, loss_recorder['sum'] / batch_size))

                loss_recorder = {
                    'cls': 0,
                    'bg': 0,
                    'sim': 0,
                    'sum': 0,
                }

                optimizer.step()
                optimizer.zero_grad()

                if learning_rate_decay: scheduler.step()

                if update_step_idx in check_points:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            save_dir,
                            'model-{}'.format(update_step_idx)))

                if update_step_idx >= max_step_num:
                    break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str)
    args = parser.parse_args()
    print(args.config_file)

    all_params = load_config_file(args.config_file)  # 加载参数
    locals().update(all_params) # 更新参数

    set_seed(1050)

    train_dataset_dict = get_dataset(dataset_name=dataset_name,
                                     subset=train_subset_name,
                                     file_paths=file_paths,
                                     sample_rate=sample_rate,
                                     base_sample_rate=base_sample_rate,
                                     feature_oversample=feature_oversample,
                                     temporal_aug=True)
    train_dataset = SingleVideoDataset(train_dataset_dict, random_select=True)  # To be checked
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=1,
                                                pin_memory=True,
                                                shuffle=True)

    train(train_loader,experiment_naming)
