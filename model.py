import torch.nn as nn
import random
from collections import OrderedDict

class BackboneNet(nn.Module):

    def __init__(self, in_features, class_num, layer_params):

        super(BackboneNet, self).__init__()

        self.class_num = class_num
        self.features = in_features
        self.embedded = nn.Sequential(nn.Conv1d(self.features,layer_params[-1][0],layer_params[-1][1], padding=1),
                                      nn.ReLU())
        self.cls = nn.Linear(layer_params[-1][0], self.class_num)
        self.att = nn.Sequential(nn.Linear(layer_params[-1][0], 1), nn.Sigmoid())

    def forward(self, input):

        embedded_feature = self.embedded(input)
        clf = self.cls(embedded_feature.transpose(1, 2))
        att = self.att(embedded_feature.transpose(1, 2))

        return clf, att

class BackboneNet1(nn.Module):

    def __init__(self, in_features, class_num,
                 layer_params):

        '''
        Layer_params:
        [[kerel_num_1, kernel_size_1],[kerel_num_2, kernel_size_2], ...]
        '''
        super(BackboneNet1, self).__init__()

        self.class_num = class_num
        self.features = in_features

        # self.dropout = nn.Dropout(p=dropout_rate)
        cls_model_list = self._get_module_list(in_features, layer_params, 'cls')
        self.cls_bottoms = nn.Sequential(OrderedDict(cls_model_list))
        self.cls_heads = nn.Linear(layer_params[-1][0], class_num)


        self.att_head = nn.Linear(layer_params[-1][0], 1)
        self.att_sigmod = nn.Sigmoid()

    def _get_module_list(self, in_features, layer_params, naming):

        module_list = []

        for layer_idx in range(len(layer_params)):

            if layer_idx == 0:
                in_chl = in_features
            else:
                in_chl = layer_params[layer_idx - 1][0]

            out_chl = layer_params[layer_idx][0]
            kernel_size = layer_params[layer_idx][1]
            conv_pad = kernel_size // 2

            module_list.append(('{}_conv_{}'.format(naming, layer_idx),
                                nn.Conv1d(in_chl,
                                          out_chl,
                                          kernel_size,
                                          padding=conv_pad)))

            module_list.append(('{}_relu_{}'.format(naming,
                                                    layer_idx), nn.ReLU()))

        return module_list

    def forward(self, x):
        # 执行的嵌入模块,以前的dropout是在全连接层上，现在是图片上，即对所有帧的某个特征一起随机失活

        base_feature = x
        # 第一步特征值平均相加

        # out：B×32×T

        cls_feature = self.cls_bottoms(base_feature)
        # in：B×T×16  out：B×T×21
        cls_feature = cls_feature.transpose(1, 2)
        cls_score = self.cls_heads(cls_feature)

        att_weight = self.att_head(cls_feature)  # in：B×T×16， out：B×T×1
        att_weight = self.att_sigmod(att_weight)

        return cls_score, att_weight



class Fusion_model(nn.Module):

    def __init__(self, in_features, class_num, dropout_rate,
                 layer_params):

        super(Fusion_model, self).__init__()

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.feature = in_features // 2
        self.model_rgb = BackboneNet(in_features=self.feature, class_num=class_num,
                                    layer_params=layer_params)
        self.model_flow = BackboneNet(in_features=self.feature, class_num=class_num,
                                    layer_params=layer_params,)

    def forward(self, input):
        input_drop = self.dropout(input.unsqueeze(3)).squeeze(3)

        rgb = input_drop[:, 0:self.feature, :]
        flow = input_drop[:, self.feature:, :]

        cas1, att1 = self.model_rgb(rgb)
        cas2, att2 = self.model_flow(flow)
        cas = (cas1 + cas2) / 2
        att = (att1 + att2) / 2

        clf_score = (cas * att).sum(dim=1) / (att.sum(dim=1) + 1)
        bg_score = (cas * (1 - att)).sum(dim=1) / ((1 - att).sum(dim=1) + 1e-6)

        out = {
            'cas':cas,
            'att':att,
            'clf':clf_score,
            'bg':bg_score
        }
        return out



class SelfNetwork(nn.Module):
    def __init__(self, in_features, class_num, dropout_rate, layer_params,):
        super(SelfNetwork, self).__init__()
        self.oriNetwork = Fusion_model(in_features, class_num,
                                       dropout_rate, layer_params)

    def forward(self, input):
        out_ori = self.oriNetwork.forward(input)

        num = random.randint(0, 11)
        input[:, :, num::12] = 0
        out_aug = self.oriNetwork.forward(input)

        return out_ori , out_aug


