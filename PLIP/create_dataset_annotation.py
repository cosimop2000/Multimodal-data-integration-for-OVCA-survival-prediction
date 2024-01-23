#run this script only AFTER having created the dataset with create_dataset_from_embedding.py

import json
import configparser
import os
import pandas as pd
import numpy as np

def get_label(file_path):
    '''
    This function uses ONLY dead patients to get the "days_to_death" label
    :param file_path: path to json file
    :return: dictionary of case_id: label
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)
        labels = {}
        for i in data:
            if 'demographic' in i.keys():
                if 'days_to_death' in i['demographic'].keys():
                    labels[i['case_id']] = i['demographic']['days_to_death']
        return labels

parser = configparser.ConfigParser()
parser.read('Data/config.ini')
file_selection = parser['embeddings']['json_file_path']
label_file = parser['embeddings']['label_path']
with open(file_selection) as json_file:  
    file_selection = json.load(json_file)
treshold = int(parser['embeddings']['treshold'])
dataset_dir = parser['embeddings']['embeddings_dataset']
class_0_folder = os.path.join(parser['embeddings']['embeddings_dataset'],'class_1')
class_1_folder = os.path.join(parser['embeddings']['embeddings_dataset'],'class_2')
case_ids = {}
#create a df three columns: filename, case_id, label


for filename in os.listdir(dataset_dir):
    if filename.endswith(('.csv')):
        svs_filename = filename[:-4] + '.svs'
        for file in file_selection:
            if file['file_name'] == svs_filename:
                case_ids[filename] = file['cases'][0]['case_id']
                break
labels = get_label(label_file)
ids_files = {}
case_paths = {}
for key in case_ids.keys():
    if case_ids[key] in labels.keys():
        if labels[case_ids[key]] < treshold:
            case_paths[os.path.join(class_0_folder, key)] = labels[case_ids[key]]
            if case_ids[key] in ids_files.keys():
                ids_files[case_ids[key]].append(os.path.join(class_0_folder, key))
            else:
                ids_files[case_ids[key]] = [os.path.join(class_0_folder, key)]

        else:
            case_paths[os.path.join(class_1_folder, key)] = labels[case_ids[key]]
            if case_ids[key] in ids_files.keys():
                ids_files[case_ids[key]].append(os.path.join(class_1_folder, key))
            else:
                ids_files[case_ids[key]] = [os.path.join(class_1_folder, key)]

with open('ids_files.json', 'w') as fp:
    json.dump(ids_files, fp)
df=pd.DataFrame.from_dict(case_paths, orient='index', columns=["daysToDeath"])

df['daysToDeath'] = np.where(df['daysToDeath'] < treshold, 0, 1)
output = parser['embeddings']['output']
df.to_csv(output, header=None, index_label=None, sep=',')
if not os.path.exists(class_0_folder):
    os.mkdir(class_0_folder)
if not os.path.exists(class_1_folder):
    os.mkdir(class_1_folder)
for row in df.iterrows():
    os.rename(os.path.join(parser['embeddings']['embeddings_dataset'], row[0].split(os.path.sep)[-1]), row[0])

        



