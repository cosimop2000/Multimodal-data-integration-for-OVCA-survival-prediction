from __future__ import print_function
import random
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
# import gbm.write_data_gbm as write_tool
import torch.optim
from dataset import dataset
import numpy as np
import torch
import copy
import torch.nn.init as init
import scipy.io as scio

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
        USE_GPU = False
        siamese_code1 = siamese_code.detach().numpy()
        siamese_code2 = copy.copy(siamese_code1[::-1])
        if USE_GPU:
            siamese_code2 = Variable(torch.from_numpy(siamese_code2)).type(torch.cuda.FloatTensor)
        else:
            siamese_code2 = Variable(torch.from_numpy(siamese_code2)).type(torch.FloatTensor)

        # Classification Loss
        classification_loss = F.cross_entropy(classification_output, target)

        # Siamese Loss
        contrastive_loss = self.contrastive_loss(siamese_code, siamese_code2, t)

        loss = contrastive_loss * 0.3 + classification_loss

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

    def forward(self, view1_specific, view1_shared, view2_specific, view2_shared, t):#, view3_specific, view3_shared, t):
        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(view1_shared, view1_specific)
        orthogonal_loss2 = self.orthogonal_loss(view2_shared, view2_specific)
        #orthogonal_loss3 = self.orthogonal_loss(view3_shared, view3_specific)
        orthogonal_loss = orthogonal_loss1 + orthogonal_loss2 #+ orthogonal_loss3

        # Siamese Loss
        contrastive_loss1 = self.contrastive_loss(view1_shared, view2_shared, t)
        # contrastive_loss3 = self.contrastive_loss(view2_shared, view3_shared, t)
        contrastive_loss = contrastive_loss1 #+ contrastive_loss3

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
        #self.specific3_l11 = nn.Linear(original_size[2], view_size[0])
        #self.specific3_l1 = nn.Linear(view_size[0], view_size[1])
        #self.specific3_l2 = nn.Linear(view_size[1], view_size[2])
        #self.specific3_l3 = nn.Linear(view_size[2], out_size)

        # no common shared layers
        self.shared1_l11 = nn.Linear(original_size[0], view_size[0])
        self.shared2_l11 = nn.Linear(original_size[1], view_size[0])
        # self.shared3_l11 = nn.Linear(original_size[2], view_size[0])

        # Common shared layers
        self.shared_l1 = nn.Linear(view_size[0], view_size[1])
        self.shared_l2 = nn.Linear(view_size[1], view_size[2])
        self.shared_l3 = nn.Linear(view_size[2], out_size)

        # Classification
        self.classification_l1 = nn.Linear(out_size * 4, c_n_units[0]) #6
        self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        self.classification_l3 = nn.Linear(c_n_units[1], 2)

        # Init weight
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal(self.specific1_l1.weight)
        init.kaiming_normal(self.specific1_l2.weight)
        init.kaiming_uniform(self.specific1_l3.weight)

        init.kaiming_uniform(self.specific2_l1.weight)
        init.kaiming_normal(self.specific2_l2.weight)
        init.kaiming_uniform(self.specific2_l3.weight)

        init.kaiming_normal(self.shared_l1.weight)
        init.kaiming_uniform(self.shared_l2.weight)
        init.kaiming_normal(self.shared_l3.weight)

        init.kaiming_uniform(self.classification_l1.weight)
        init.kaiming_uniform(self.classification_l2.weight)
        init.kaiming_uniform(self.classification_l3.weight)

    def classify_code(self, code):  # code为拼接起来后的表示，进行分类
        classification_output = F.relu(self.classification_l1(F.dropout(code)))
        classification_output = F.relu(self.classification_l2(F.dropout(classification_output)))
        classification_output = F.relu(self.classification_l3(classification_output))

        return classification_output

    def forward(self, view1_input, view2_input): #view3_input
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
        #view3_specific11 = F.relu(self.specific3_l11(view3_input))
        #view3_specific1 = F.relu(self.specific3_l1(view3_specific11))
        #view3_specific2 = F.relu(self.specific3_l2(view3_specific1))
        #view3_specific3 = F.relu(self.specific3_l3(view3_specific2))

        #view3_shared11 = F.relu(self.shared3_l11(view3_input))
        #view3_shared1 = F.relu(self.shared_l1(view3_shared11))
        #view3_shared2 = F.relu(self.shared_l2(view3_shared1))
        #view3_shared3 = F.relu(self.shared_l3(view3_shared2))

        # Classification
        classification_input = torch.cat([view1_specific3, view1_shared3, view2_shared3, view2_specific3], dim=1)
        siamese_code = classification_input
        classification_output = self.classify_code(classification_input)

        return view1_specific3, view1_shared3, view2_specific3, view2_shared3, \
                siamese_code, classification_output


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
        #self.shared31 = nn.Linear(view_size[2], n_unit[2])
        #self.shared32 = nn.Linear(n_unit[2], out_size)
        #self.specific31 = nn.Linear(view_size[2], n_unit[2])
        #self.specific32 = nn.Linear(n_unit[2], out_size)

    def forward(self, view1_input, view2_input):#, view3_input):
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
        #view3_specific1 = F.relu(self.specific31(view3_input))
        #view3_specific = F.relu(self.specific32(view3_specific1))
        #view3_shared1 = F.relu(self.shared31(view3_input))
        #view3_shared = F.relu(self.shared32(view3_shared1))

        return view1_specific, view1_shared, view2_specific, view2_shared#, view3_specific, view3_shared


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
        #self.shared3 = nn.Linear(view_size[0], view_size[1])
        #self.specific3 = nn.Linear(view_size[0], view_size[1])

    def forward(self, input_view1_specific, input_view1_shared, input_view2_specific, input_view2_shared): #, ccc
                #input_view3_specific, input_view3_shared):
        # View1
        view1_specific = F.relu(self.specific1(input_view1_specific))
        view1_shared = F.relu(self.shared1(input_view1_shared))

        # View2
        view2_specific = F.relu(self.specific2(input_view2_specific))
        view2_shared = F.relu(self.shared2(input_view2_shared))

        # View3
        #view3_specific = F.relu(self.specific3(input_view3_specific))
        #view3_shared = F.relu(self.shared3(input_view3_shared))

        return view1_specific, view1_shared, view2_specific, view2_shared #, view3_specific, view3_shared


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
        #self.shared3 = nn.Linear(view_size[0], view_size[1])
        #self.specific3 = nn.Linear(view_size[0], view_size[1])

    def forward(self, input_view1_specific, input_view1_shared, input_view2_specific, input_view2_shared):#,
                #input_view3_specific, input_view3_shared):
        # View1
        view1_specific = F.relu(self.specific1(input_view1_specific))
        view1_shared = F.relu(self.shared1(input_view1_shared))

        # View2
        view2_specific = F.relu(self.specific2(input_view2_specific))
        view2_shared = F.relu(self.shared2(input_view2_shared))

        # View3
        #view3_specific = F.relu(self.specific3(input_view3_specific))
        #view3_shared = F.relu(self.shared3(input_view3_shared))

        return view1_specific, view1_shared, view2_specific, view2_shared#, view3_specific, view3_shared


def main():
    # Hyper Parameters
    EPOCH = 138  # 138 train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 148
    USE_GPU = False

    # Load data

    train_size = 0.8
    test_size = 1 - train_size
    data = dataset()
    #split data in train, test and validation datasets
    train_data, test_data = torch.utils.data.random_split(data, [int(train_size*len(data))+1, int(test_size*len(data))])

    # Build Model
    model = SiameseSharedAndSpecificClassifier(original_size=[60660, 25310], view_size=[1024, 256, 128], out_size=32, c_n_units=[64, 32])
    SharedSpecific1 = SharedAndSpecific1(view_size=[60660, 25310], n_unit=[1024, 1024], out_size=256)
    SharedSpecific2 = SharedAndSpecific2(view_size=[256, 128])
    SharedSpecific3 = SharedAndSpecific3(view_size=[128, 32])
    # print(model)
    if USE_GPU:
        model = model.cuda()
        SharedSpecific1 = SharedSpecific1.cuda()
        SharedSpecific2 = SharedSpecific2.cuda()
        SharedSpecific3 = SharedSpecific3.cuda()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = SiameseSharedAndSpecificLoss(margin=2.0)

    SharedSpecific1_optimizer = torch.optim.Adam(SharedSpecific1.parameters())
    SharedSpecific1_loss = SharedAndSpecificLoss(margin=2.0)

    SharedSpecific2_optimizer = torch.optim.Adam(SharedSpecific2.parameters())
    SharedSpecific2_loss = SharedAndSpecificLoss(margin=2.0)

    SharedSpecific3_optimizer = torch.optim.Adam(SharedSpecific3.parameters())
    SharedSpecific3_loss = SharedAndSpecificLoss(margin=2.0)

    ## begin training==========================================================================
    print("Training...")
    train_loss_ = []
    train_acc_ = []
    train_pre_ = []
    train_auc_ = []
    train_recall_ = []
    test_acc_ = []
    test_pre_ = []
    test_auc_ = []
    test_recall_ = []
    # Data Loader for easy mini-batch return in training
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=30, shuffle=True)
    for epoch in range(EPOCH):

        # training epoch
        total_loss = 0.0
        total = 0.0
        train_count = 0.0
        test_count = 0.0
        train_acc = 0.0
        train_auc = 0.0
        train_pre = 0.0
        train_recall = 0.0
        test_acc = 0.0
        test_auc = 0.0
        test_pre = 0.0
        test_recall = 0.0
        for iter, traindata in enumerate(train_loader):
            batch_view1, batch_view2, label = traindata  #batch_view3

            # Siamese input
            batch_view1 = batch_view1.numpy()
            batch_view2 = batch_view2.numpy()
            # batch_view3 = batch_view3.numpy()
            train_labels = np.squeeze(label.numpy())

            t_train_labels = copy.copy(train_labels[::-1])
            batch_t = np.array(train_labels == t_train_labels, dtype=np.longlong)
            batch_shared_t = np.array(train_labels == train_labels, dtype=np.longlong)

            # To Variable
            if USE_GPU:
                batch_view1 = Variable(torch.from_numpy(batch_view2)).type(torch.cuda.FloatTensor)
                batch_view2 = Variable(torch.from_numpy(batch_view2)).type(torch.cuda.FloatTensor)
                #batch_view3 = Variable(torch.from_numpy(batch_view3)).type(torch.cuda.FloatTensor)
                train_labels = Variable(torch.from_numpy(train_labels)).type(torch.cuda.LongTensor)
                batch_t = Variable(torch.from_numpy(batch_t)).type(torch.cuda.FloatTensor)
                batch_shared_t = Variable(torch.from_numpy(batch_shared_t)).type(torch.cuda.FloatTensor)
            else:
                batch_view1 = Variable(torch.from_numpy(batch_view1)).type(torch.FloatTensor)
                batch_view2 = Variable(torch.from_numpy(batch_view2)).type(torch.FloatTensor)
                #batch_view3 = Variable(torch.from_numpy(batch_view3)).type(torch.FloatTensor)
                train_labels = Variable(torch.from_numpy(train_labels)).type(torch.LongTensor)
                batch_t = Variable(torch.from_numpy(batch_t)).type(torch.FloatTensor)
                batch_shared_t = Variable(torch.from_numpy(batch_shared_t)).type(torch.FloatTensor)

            # Train Multiple group network 1 ==========================================================================
            SharedSpecific1_optimizer.zero_grad()
            batch_view1_specific1, batch_view1_shared1, batch_view2_specific1, batch_view2_shared1, \
                = SharedSpecific1(view1_input=batch_view1.detach(), view2_input=batch_view2.detach()) #,
                                  #view3_input=batch_view3.detach())

            comsp_loss1 = SharedSpecific1_loss(view1_specific=batch_view1_specific1, view1_shared=batch_view1_shared1,
                                               view2_specific=batch_view2_specific1, view2_shared=batch_view2_shared1,
                                               t=batch_shared_t)

            comsp_loss1.backward()
            SharedSpecific1_optimizer.step()

            # Train Multiple group network 2 ==========================================================================
            SharedSpecific2_optimizer.zero_grad()
            batch_view1_specific2, batch_view1_shared2, batch_view2_specific2, batch_view2_shared2, \
                = SharedSpecific2(input_view1_specific=batch_view1_specific1.detach(),
                                  input_view1_shared=batch_view1_shared1.detach(),
                                  input_view2_specific=batch_view2_specific1.detach(),
                                  input_view2_shared=batch_view2_shared1.detach())

            comsp_loss2 = SharedSpecific2_loss(view1_specific=batch_view1_specific2, view1_shared=batch_view1_shared2,
                                               view2_specific=batch_view2_specific2, view2_shared=batch_view2_shared2,
                                               t=batch_shared_t)

            comsp_loss2.backward()
            SharedSpecific2_optimizer.step()

            # Train Multiple group network 3 ==========================================================================
            SharedSpecific3_optimizer.zero_grad()
            batch_view1_specific3, batch_view1_shared3, batch_view2_specific3, batch_view2_shared3, \
                = SharedSpecific3(input_view1_specific=batch_view1_specific2.detach(),
                                  input_view1_shared=batch_view1_shared2.detach(),
                                  input_view2_specific=batch_view2_specific2.detach(),
                                  input_view2_shared=batch_view2_shared2.detach())

            comsp_loss3 = SharedSpecific3_loss(view1_specific=batch_view1_specific3, view1_shared=batch_view1_shared3,
                                               view2_specific=batch_view2_specific3, view2_shared=batch_view2_shared3,
                                               t=batch_shared_t)

            comsp_loss3.backward()
            SharedSpecific3_optimizer.step()

            # model, generator Loss =======================================================================
            # Should detach to avoid to train the Discriminator, so just don't update the discriminator, only
            # update the generator

            optimizer.zero_grad()
            batch_view1_specific, batch_view1_shared, batch_view2_specific, batch_view2_shared,\
            batch_siamese_code, batch_classification_output \
                = model(batch_view1, batch_view2)

            loss = loss_function(siamese_code=batch_siamese_code, classification_output=batch_classification_output,
                                 target=train_labels, t=batch_t)
            loss.backward()
            optimizer.step()

            # scio.savemat('E:/Data/GBM_all.mat', {'GBM_all': batch_classification_output.detach().cpu().numpy()})
            # scio.savemat('E:/Data/GBM_all_label.mat', {'GBM_all_label': train_labels.detach().cpu().numpy()})

            #scio.savemat('E:/Data/GBM_specific1.mat', {'GBM_specific1': batch_view1_specific.detach().cpu().numpy()})
            #scio.savemat('E:/Data/GBM_specific2.mat', {'GBM_specific2': batch_view2_specific.detach().cpu().numpy()})
            #scio.savemat('E:/Data/GBM_shared1.mat', {'GBM_shared1': batch_view1_shared.detach().cpu().numpy()})
            #scio.savemat('E:/Data/GBM_shared2.mat', {'GBM_shared2': batch_view2_shared.detach().cpu().numpy()})
#
            #scio.savemat('E:/Data/GBM_component_label.mat', {'GBM_component_label': train_labels.detach().cpu().numpy()})

            # calculation training acc
            total += len(train_labels)
            total_loss += loss.item()
            train_count += 1
            acc, pre, auc, recall = measure_performance(batch_classification_output, train_labels)
            train_acc += acc
            train_auc += auc
            train_pre += pre
            train_recall += recall

        train_acc_.append(train_acc/train_count)
        train_pre_.append(train_pre/train_count)
        train_auc_.append(train_auc/train_count)
        train_recall_.append(train_recall/train_count)
        train_loss_.append(total_loss / total)

        # testing epoch ========================================================================================
        for iter, testdata in enumerate(test_loader):
            test_view1_inputs, test_view2_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)

            if USE_GPU:
                test_view1_inputs, test_view2_inputs, test_labels \
                    = Variable(test_view1_inputs.cuda()), \
                      Variable(test_view2_inputs.cuda()), \
                       test_labels.cuda()
            else:
                test_view1_inputs = Variable(test_view1_inputs).type(torch.FloatTensor)
                test_view2_inputs = Variable(test_view2_inputs).type(torch.FloatTensor)
                test_labels = Variable(test_labels).type(torch.LongTensor)

            view1_specific, view1_shared, view2_specific, view2_shared,\
            siamese_code, classification_output = \
                model.forward(test_view1_inputs, test_view2_inputs)

            # calculation testing acc
            _, predicted = torch.max(classification_output.data, 1)
            test_count += 1
            acc, pre, auc, recall = measure_performance(classification_output, test_labels)
            test_acc += acc
            test_auc += auc
            test_pre += pre
            test_recall += recall
        test_acc_.append(test_acc/test_count)
        test_pre_.append(test_pre/test_count)
        test_auc_.append(test_auc/test_count)
        test_recall_.append(test_recall/test_count)
		
		# scio.savemat('E:/Data/GBM_Predict.mat', {'GBM_Predict': classification_output.detach().cpu().numpy()})
        # scio.savemat('E:/Data/GBM_Label.mat', {'GBM_Label': test_labels.detach().cpu().numpy()})



        print('[Epoch: %3d/%3d] Training Loss: %.3f,Training Acc: %.3f, Training Pre: %.3f, Training Auc: %.3f, Training Recall: %.3f' %
              (epoch, EPOCH, train_loss_[epoch], train_acc_[epoch], train_pre_[epoch], train_auc_[epoch], train_recall_[epoch]))

        print('test Acc: %.3f, test Pre: %.3f, test Auc: %.3f, test Recall: %.3f' %
            (test_acc_[epoch], test_pre_[epoch], test_auc_[epoch], test_recall_[epoch]))



if __name__ == "__main__":
    main()


