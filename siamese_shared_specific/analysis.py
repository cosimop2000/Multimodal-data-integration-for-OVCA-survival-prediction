import json
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with open('results_ttv.json') as json_file:
        results = json.load(json_file)
        acc = results['acc']
        auc = results['auc']
        recall = results['recall']
        pre = results['pre']
        #Box plot of acc pre recall auc
        data = [acc, pre, recall, auc]
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data)
        ax.set_xticklabels(['acc', 'pre', 'recall', 'auc'])
        plt.title('Boxplot of acc pre recall auc')
        plt.savefig('boxplot.png')
        plt.show()
        #Bar plot of acc pre recall auc
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        acc_mean = np.mean(acc)
        pre_mean = np.mean(pre)
        recall_mean = np.mean(recall)
        auc_mean = np.mean(auc)
        acc_std = np.std(acc)
        pre_std = np.std(pre)
        recall_std = np.std(recall)
        auc_std = np.std(auc)
        means = [acc_mean, pre_mean, recall_mean, auc_mean]
        stds = [acc_std, pre_std, recall_std, auc_std]
        ax.bar(['acc', 'pre', 'recall', 'auc'], means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
        plt.title('Bar plot of acc pre recall auc')
        plt.savefig('barplot.png')
        plt.show()

