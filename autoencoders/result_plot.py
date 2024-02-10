import pandas as pd
import configparser, os, json
import matplotlib.pyplot as plt

parser = configparser.ConfigParser()
parser.read('Data/config.ini')
base_path = parser['autoencoder']['base_path']   

f = open(os.path.join(base_path, "results.json"))
results_list = json.load(f)
average_results = {}

for results in results_list:
    for metric, value in results.items():
        if metric in average_results:
            average_results[metric] += value
        else:
            average_results[metric] = value

for metric, value in average_results.items():
    average_results[metric] = value / len(results_list)
    
print(average_results)

acc = [results['Accuracy'] for results in results_list]
auc =[results['AUC'] for results in results_list]
recall = [results['Recall'] for results in results_list]
pre = [results['Precision'] for results in results_list]
#Box plot of acc pre recall auc
data = [acc, pre, recall, auc]
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['acc', 'pre', 'recall', 'auc'])
plt.title('Boxplot of acc pre recall auc')
plt.savefig('boxplot.png')
plt.show()