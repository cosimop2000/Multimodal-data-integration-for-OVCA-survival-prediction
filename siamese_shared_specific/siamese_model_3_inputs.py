from __future__ import print_function
import random
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
# import gbm.write_data_gbm as write_tool
import torch.optim
import numpy as np
import torch
import copy
import torch.nn.init as init

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

setup_seed(9)

def measure_performance(output, target):
    """accuracy"""
    with torch.no_grad():
        auc = 0  # if only one class in this batch, auc = 0
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        pred = pred.cpu().data.numpy()[0]
        target = target.cpu().data.numpy()
        acc = metrics.accuracy_score(target, pred)
        pre = metrics.precision_score(target, pred, pos_label=0)
        recall = metrics.recall_score(target, pred, pos_label=0)
        auc = metrics.roc_auc_score(target, pred)
        return acc, pre, auc, recall


class SiameseSharedAndSpecificLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(SiameseSharedAndSpecificLoss, self).__init__()
        self.margin = margin

    # 对比损失
    def contrastive_loss(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def forward(self, siamese_code, classification_output, target, t):
        USE_GPU = True
        siamese_code1 = siamese_code.detach().cpu().numpy()
        siamese_code2 = copy.copy(siamese_code1[::-1])
        if USE_GPU:
            siamese_code2 = Variable(torch.from_numpy(siamese_code2)).type(torch.cuda.FloatTensor)
            
        else:
            siamese_code2 = Variable(torch.from_numpy(siamese_code2)).type(torch.FloatTensor)

        # Classification Loss
        classification_loss = F.cross_entropy(classification_output, target)
        #print(f'#############CLASSIFICATION LOSS:{classification_loss}################')

        # Siamese Loss
        contrastive_loss = self.contrastive_loss(siamese_code, siamese_code2, t)

        loss = classification_loss# + contrastive_loss*0.01

        return loss


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(SharedAndSpecificLoss, self).__init__()
        self.margin = margin

    # 正交损失
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.t().matmul(specific)
        cost = correlation_matrix.matmul(correlation_matrix).mean()
        cost = F.relu(cost)
        return cost

    # 对比损失GBMValidation
    def contrastive_loss(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def forward(self, view1_specific, view1_shared, view2_specific, view2_shared, view3_specific, view3_shared, t):
        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(view1_shared, view1_specific)
        orthogonal_loss2 = self.orthogonal_loss(view2_shared, view2_specific)
        orthogonal_loss3 = self.orthogonal_loss(view3_shared, view3_specific)
        orthogonal_loss = orthogonal_loss1 + orthogonal_loss2 + orthogonal_loss3

        # Siamese Loss
        contrastive_loss1 = self.contrastive_loss(view1_shared, view2_shared, t)
        contrastive_loss3 = self.contrastive_loss(view2_shared, view3_shared, t)
        contrastive_loss = contrastive_loss1 + contrastive_loss3

        loss = contrastive_loss + orthogonal_loss

        return loss


class SiameseSharedAndSpecificClassifier(nn.Module):
    def __init__(self, original_size=[495, 472, 457], view_size=[457, 256, 128], out_size=32, c_n_units=[64, 32]):
        super(SiameseSharedAndSpecificClassifier, self).__init__()

        # View1 specific
        self.specific1_l11 = nn.Linear(original_size[0], view_size[0])
        self.specific1_l1 = nn.Linear(view_size[0], view_size[1])
        self.specific1_l2 = nn.Linear(view_size[1], view_size[2])
        self.specific1_l3 = nn.Linear(view_size[2], out_size)

        # View2 specific
        self.specific2_l11 = nn.Linear(original_size[1], view_size[0])
        self.specific2_l1 = nn.Linear(view_size[0], view_size[1])
        self.specific2_l2 = nn.Linear(view_size[1], view_size[2])
        self.specific2_l3 = nn.Linear(view_size[2], out_size)

        # View3 specific
        self.specific3_l11 = nn.Linear(original_size[2], view_size[0])
        self.specific3_l1 = nn.Linear(view_size[0], view_size[1])
        self.specific3_l2 = nn.Linear(view_size[1], view_size[2])
        self.specific3_l3 = nn.Linear(view_size[2], out_size)

        # no common shared layers
        self.shared1_l11 = nn.Linear(original_size[0], view_size[0])
        self.shared2_l11 = nn.Linear(original_size[1], view_size[0])
        self.shared3_l11 = nn.Linear(original_size[2], view_size[0])

        # Common shared layers
        self.shared_l1 = nn.Linear(view_size[0], view_size[1])
        self.shared_l2 = nn.Linear(view_size[1], view_size[2])
        self.shared_l3 = nn.Linear(view_size[2], out_size)

        # Classification
        self.classification_l1 = nn.Linear(out_size * 6, c_n_units[0]) #6
        self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        self.classification_l3 = nn.Linear(c_n_units[1], 2)

        # Init weight
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.specific1_l1.weight)
        init.kaiming_normal_(self.specific1_l2.weight)
        init.kaiming_uniform_(self.specific1_l3.weight)

        init.kaiming_uniform_(self.specific2_l1.weight)
        init.kaiming_normal_(self.specific2_l2.weight)
        init.kaiming_uniform_(self.specific2_l3.weight)

        init.kaiming_normal_(self.shared_l1.weight)
        init.kaiming_uniform_(self.shared_l2.weight)
        init.kaiming_normal_(self.shared_l3.weight)

        init.kaiming_uniform_(self.classification_l1.weight)
        init.kaiming_uniform_(self.classification_l2.weight)
        init.kaiming_uniform_(self.classification_l3.weight)

    def classify_code(self, code):  # code为拼接起来后的表示，进行分类
        classification_output = F.relu(self.classification_l1(F.dropout(code)))
        classification_output = F.relu(self.classification_l2(F.dropout(classification_output)))
        classification_output = self.classification_l3(classification_output)

        return classification_output

    def forward(self, view1_input, view2_input, view3_input): #view3_input
        # View1
        view1_specific11 = F.relu(self.specific1_l11(view1_input))
        view1_specific1 = F.relu(self.specific1_l1(view1_specific11))
        view1_specific2 = F.relu(self.specific1_l2(view1_specific1))
        view1_specific3 = F.relu(self.specific1_l3(view1_specific2))

        view1_shared11 = F.relu(self.shared1_l11(view1_input))
        view1_shared1 = F.relu(self.shared_l1(view1_shared11))
        view1_shared2 = F.relu(self.shared_l2(view1_shared1))
        view1_shared3 = F.relu(self.shared_l3(view1_shared2))

        # View2
        view2_specific11 = F.relu(self.specific2_l11(view2_input))
        view2_specific1 = F.relu(self.specific2_l1(view2_specific11))
        view2_specific2 = F.relu(self.specific2_l2(view2_specific1))
        view2_specific3 = F.relu(self.specific2_l3(view2_specific2))

        view2_shared11 = F.relu(self.shared2_l11(view2_input))
        view2_shared1 = F.relu(self.shared_l1(view2_shared11))
        view2_shared2 = F.relu(self.shared_l2(view2_shared1))
        view2_shared3 = F.relu(self.shared_l3(view2_shared2))

        # View3
        view3_specific11 = F.relu(self.specific3_l11(view3_input))
        view3_specific1 = F.relu(self.specific3_l1(view3_specific11))
        view3_specific2 = F.relu(self.specific3_l2(view3_specific1))
        view3_specific3 = F.relu(self.specific3_l3(view3_specific2))

        view3_shared11 = F.relu(self.shared3_l11(view3_input))
        view3_shared1 = F.relu(self.shared_l1(view3_shared11))
        view3_shared2 = F.relu(self.shared_l2(view3_shared1))
        view3_shared3 = F.relu(self.shared_l3(view3_shared2))

        # Classification
        classification_input = torch.cat([view1_specific3, view1_shared3, view2_shared3, view2_specific3, view3_shared3, view3_specific3], dim=1)
        siamese_code = classification_input
        classification_output = self.classify_code(classification_input)

        return view1_specific3, view1_shared3, view2_specific3, view2_shared3, \
               view3_specific3, view3_shared3, siamese_code, classification_output


class SharedAndSpecific1(nn.Module):
    def __init__(self, view_size=[495, 472, 457], n_unit=[457, 457, 457], out_size=256):
        super(SharedAndSpecific1, self).__init__()

        # View1
        self.shared11 = nn.Linear(view_size[0], n_unit[0])
        self.shared12 = nn.Linear(n_unit[0], out_size)
        self.specific11 = nn.Linear(view_size[0], n_unit[0])
        self.specific12 = nn.Linear(n_unit[0], out_size)

        # View2
        self.shared21 = nn.Linear(view_size[1], n_unit[1])
        self.shared22 = nn.Linear(n_unit[1], out_size)
        self.specific21 = nn.Linear(view_size[1], n_unit[1])
        self.specific22 = nn.Linear(n_unit[1], out_size)

        # View3
        self.shared31 = nn.Linear(view_size[2], n_unit[2])
        self.shared32 = nn.Linear(n_unit[2], out_size)
        self.specific31 = nn.Linear(view_size[2], n_unit[2])
        self.specific32 = nn.Linear(n_unit[2], out_size)

    def forward(self, view1_input, view2_input, view3_input):
        # View1
        view1_specific1 = F.relu(self.specific11(view1_input))
        view1_specific = F.relu(self.specific12(view1_specific1))
        view1_shared1 = F.relu(self.shared11(view1_input))
        view1_shared = F.relu(self.shared12(view1_shared1))

        # View2
        view2_specific1 = F.relu(self.specific21(view2_input))
        view2_specific = F.relu(self.specific22(view2_specific1))
        view2_shared1 = F.relu(self.shared21(view2_input))
        view2_shared = F.relu(self.shared22(view2_shared1))

        # View3
        view3_specific1 = F.relu(self.specific31(view3_input))
        view3_specific = F.relu(self.specific32(view3_specific1))
        view3_shared1 = F.relu(self.shared31(view3_input))
        view3_shared = F.relu(self.shared32(view3_shared1))

        return view1_specific, view1_shared, view2_specific, view2_shared, view3_specific, view3_shared


class SharedAndSpecific2(nn.Module):
    def __init__(self, view_size=[3000, 1840]):
        super(SharedAndSpecific2, self).__init__()

        # View1
        self.shared1 = nn.Linear(view_size[0], view_size[1])
        self.specific1 = nn.Linear(view_size[0], view_size[1])

        # View2
        self.shared2 = nn.Linear(view_size[0], view_size[1])
        self.specific2 = nn.Linear(view_size[0], view_size[1])

        # View3
        self.shared3 = nn.Linear(view_size[0], view_size[1])
        self.specific3 = nn.Linear(view_size[0], view_size[1])

    def forward(self, input_view1_specific, input_view1_shared, input_view2_specific, input_view2_shared, input_view3_specific, input_view3_shared):
        # View1
        view1_specific = F.relu(self.specific1(input_view1_specific))
        view1_shared = F.relu(self.shared1(input_view1_shared))

        # View2
        view2_specific = F.relu(self.specific2(input_view2_specific))
        view2_shared = F.relu(self.shared2(input_view2_shared))

        # View3
        view3_specific = F.relu(self.specific3(input_view3_specific))
        view3_shared = F.relu(self.shared3(input_view3_shared))

        return view1_specific, view1_shared, view2_specific, view2_shared, view3_specific, view3_shared


class SharedAndSpecific3(nn.Module):
    def __init__(self, view_size=[3000, 1840]):
        super(SharedAndSpecific3, self).__init__()

        # View1
        self.shared1 = nn.Linear(view_size[0], view_size[1])
        self.specific1 = nn.Linear(view_size[0], view_size[1])

        # View2
        self.shared2 = nn.Linear(view_size[0], view_size[1])
        self.specific2 = nn.Linear(view_size[0], view_size[1])

        # View3
        self.shared3 = nn.Linear(view_size[0], view_size[1])
        self.specific3 = nn.Linear(view_size[0], view_size[1])

    def forward(self, input_view1_specific, input_view1_shared, input_view2_specific, input_view2_shared, input_view3_specific, input_view3_shared):
        # View1
        view1_specific = F.relu(self.specific1(input_view1_specific))
        view1_shared = F.relu(self.shared1(input_view1_shared))

        # View2
        view2_specific = F.relu(self.specific2(input_view2_specific))
        view2_shared = F.relu(self.shared2(input_view2_shared))

        # View3
        view3_specific = F.relu(self.specific3(input_view3_specific))
        view3_shared = F.relu(self.shared3(input_view3_shared))

        return view1_specific, view1_shared, view2_specific, view2_shared, view3_specific, view3_shared