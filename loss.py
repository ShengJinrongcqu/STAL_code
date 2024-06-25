import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

class Losses(nn.Module):
    def __init__(self, class_num, loss_bg_weight, loss_sim_weight):
        super(Losses, self).__init__()

        self.class_num = class_num
        self.loss_bg_weight = loss_bg_weight
        self.loss_sim_weight = loss_sim_weight
        self.loss_recoder = {}

    def loss_entropy(self, x, y, weight = 1):
        gt_prob = y / y.sum(dim=1, keepdim=True)
        return -(gt_prob * F.log_softmax(x, dim=1)).sum(dim=1).mean() * weight

    def loss_classification(self, out_ori, cls_label):
        label = torch.zeros(out_ori['clf'].shape).cuda()
        label[:, cls_label] = 1.0
        bg_label = torch.cat([torch.zeros((1, self.class_num - 1)), torch.ones((1, 1))], dim=1).cuda()

        loss_fg = self.loss_entropy(out_ori['clf'], label)  # + loss_fg(clf_score2, label)
        loss_bg = (self.loss_entropy(out_ori['bg'], bg_label)) * self.loss_bg_weight
        w_clf = torch.exp(-2.5 * (F.softmax(out_ori['clf'].clone().detach(), dim=1)[:, cls_label].sum(dim=1)))
        w_clf = torch.pow(1-(F.softmax(clf_score.clone().detach(), dim=1)[:, labels].sum(dim=1)), 0.5)

        self.loss_recoder['clf'] = loss_fg.item()
        self.loss_recoder['bg'] = loss_bg.item()

        return loss_fg + loss_bg

    def loss_similarity(self,out_ori, out_aug):
        cas_label = F.softmax(out_ori['cas'], dim=2).clone().detach().squeeze(0)
        att_label = self.get_gaussian_label(out_ori['att'])

        loss_att = (out_aug['att'] - att_label).abs().mean()
        loss_cas = self.loss_entropy(out_aug['cas'].squeeze(0), cas_label)
        lambda_ = loss_att.clone().detach() / loss_cas.clone().detach()
        loss_sim = (loss_att + loss_cas * lambda_) * self.loss_sim_weight

        self.loss_recoder['sim'] = loss_sim.item()

        return loss_sim

    def get_gaussian_label(self, score):
        # 高斯平滑
        score = score.clone().detach().cpu().numpy()
        score = gaussian_filter1d(score, sigma=1, axis=1)
        score = torch.Tensor(score).cuda()
        return score

    def forward(self, out_ori, out_aug, cls_label):

        loss = self.loss_classification(out_ori, cls_label)\
               +self.loss_similarity(out_ori, out_aug)

        return  loss, self.loss_recoder

