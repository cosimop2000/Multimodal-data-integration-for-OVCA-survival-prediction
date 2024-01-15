from __future__ import print_function
from torch.autograd import Variable
# import gbm.write_data_gbm as write_tool
import torch.optim
from dataset import dataset
import numpy as np
import torch
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from siamese_model import *
def main():
    # Hyper Parameters
    EPOCH = 25  # 138 train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = 300
    USE_GPU = True

    # Load data

    train_size = 0.8
    test_size = 1 - train_size
    data = dataset()
    #split data in train, test and validation datasets
    train_data, test_data = torch.utils.data.random_split(data, [int(train_size*len(data))+1, int(test_size*len(data))])

    # Build Model
    model = SiameseSharedAndSpecificClassifier(original_size=[train_data[0][0].shape[0], train_data[0][1].shape[0]], view_size=[1024, 256, 128], out_size=32, c_n_units=[64, 32])
    SharedSpecific1 = SharedAndSpecific1(view_size=[train_data[0][0].shape[0], train_data[0][1].shape[0]], n_unit=[1024, 1024], out_size=256)
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
    loss_function = SiameseSharedAndSpecificLoss(margin=2.)

    SharedSpecific1_optimizer = torch.optim.Adam(SharedSpecific1.parameters())
    SharedSpecific1_loss = SharedAndSpecificLoss(margin=2.)

    SharedSpecific2_optimizer = torch.optim.Adam(SharedSpecific2.parameters())
    SharedSpecific2_loss = SharedAndSpecificLoss(margin=2.)

    SharedSpecific3_optimizer = torch.optim.Adam(SharedSpecific3.parameters())
    SharedSpecific3_loss = SharedAndSpecificLoss(margin=2.)

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
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=59, shuffle=True)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    total_loss = 1e10
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
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
                batch_view1 = Variable(torch.from_numpy(batch_view1)).type(torch.cuda.FloatTensor)
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
            
            #perform tsne on the siamese code to reduce it to 2 dimensions
            siamese_code_np = batch_siamese_code.detach().cpu().numpy()

            # Create a TSNE object
            tsne = TSNE(n_components=2, random_state=0)
            #plot the tsne, use the labels to color the points
            if epoch % 10 == 0:
               low_dim_data = tsne.fit_transform(siamese_code_np)
               plt.figure(figsize=(10, 5))
               plt.subplot(121)
               plt.title('train')
               plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=train_labels.detach().cpu().numpy())
               plt.show()
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
        lr_scheduler.step()
        train_acc_.append(train_acc/train_count)
        train_pre_.append(train_pre/train_count)
        train_auc_.append(train_auc/train_count)
        train_recall_.append(train_recall/train_count)
        train_loss_.append(total_loss / train_count)

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
            tsne = TSNE(n_components=2, random_state=0, perplexity=30)
            if epoch % 10 == 0:
               low_dim_data = tsne.fit_transform(siamese_code.detach().cpu().numpy())
               plt.figure(figsize=(10, 5))
               plt.subplot(121)
               plt.title('test')
               plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=test_labels.detach().cpu().numpy())
               plt.show()
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


        
        print('[Epoch: %3d/%3d] Training Loss: %.3f,Training Acc: %.3f, Training Pre: %.3f, Training Auc: %.3f, Training Recall: %.3f, LR: %.5f' %
              (epoch, EPOCH, train_loss_[epoch], train_acc_[epoch], train_pre_[epoch], train_auc_[epoch], train_recall_[epoch], optimizer.param_groups[0]['lr']))

        print('test Acc: %.3f, test Pre: %.3f, test Auc: %.3f, test Recall: %.3f' %
            (test_acc_[epoch], test_pre_[epoch], test_auc_[epoch], test_recall_[epoch]))



if __name__ == "__main__":
    main()


