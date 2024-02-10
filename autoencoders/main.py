
from src.network.customics import CustOMICS
from src.tools.prepare_dataset import prepare_dataset
from src.tools.utils import get_sub_omics_df, get_common_samples
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import json


import json
from datetime import datetime


import os
import configparser


parser = configparser.ConfigParser()
parser.read('Data/config.ini')
base_path = parser['autoencoder']['base_path']   


omics_df = {
            'gene_exp': pd.read_csv(os.path.join(base_path, 'TCGA-OV.htseq_fpkm-uq.tsv'), sep='\t', index_col=0, header=0).T,
            'methyl': pd.read_csv(os.path.join(base_path, 'TCGA-OV.methylation27.tsv'), sep='\t', index_col=0, header=0).T,
            #'cnv': pd.read_csv(os.path.join(base_path, 'TCGA-OV.gistic.tsv'), sep='\t', index_col=0, header=0).T,
            }
clinical_df = pd.read_csv(os.path.join(base_path, 'TCGA-OV.survival_labeled.tsv'), sep='\t', index_col=0, header=0)


for name, omics in omics_df.items():
    omics.dropna(inplace=True, how='all', axis=1)
    omics.fillna(value=0, inplace=True)   
    print(omics.shape)


lt_samples = get_common_samples([*list(omics_df.values()), clinical_df])
print(f"n of sample: {len(lt_samples)}")

RUNS = 20
result_list = []


batch_size = 32
n_epochs = 20
device = torch.device('cpu')
label = 'OS.label'
event = 'OS'
surv_time = 'OS.time'

task = 'classification'
sources = omics_df.keys()

hidden_dim = [512, 256]
central_dim = [512, 256]
rep_dim = 256
latent_dim = 256
num_classes = 2
dropout = 0.45
beta = 1
lambda_classif = 5
classifier_dim = [128]
lambda_survival = 0
survival_dim = [256]



for i in range(RUNS):
    samples_train, samples_test = train_test_split(lt_samples, test_size=0.2)
    samples_train, samples_val = train_test_split(samples_train, test_size=0.2)

    omics_train = get_sub_omics_df(omics_df, samples_train)
    omics_val = get_sub_omics_df(omics_df, samples_val)
    omics_test = get_sub_omics_df(omics_df, samples_test)


    x_dim = [omics_df[omic_source].shape[1] for omic_source in omics_df.keys()]
    
    source_params = {}
    central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim, 'norm': True, 'dropout': dropout, 'beta': beta}
    classif_params = {'n_class': num_classes, 'lambda': lambda_classif, 'hidden_layers': classifier_dim, 'dropout': dropout}
    surv_params = {'lambda': lambda_survival, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-3, 'norm': True, 'dropout': dropout}
    for i, source in enumerate(sources):
        source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim, 'norm': True, 'dropout': 0.2}
    train_params = {'switch': 5, 'lr': 1e-3}

    model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                            surv_params=surv_params, train_params=train_params, device=device).to(device)
    print('Number of Parameters: ', model.get_number_parameters())


    model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=True)


    results = model.evaluate(omics_test=omics_test, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                    task=task, batch_size=32, plot_roc=True)

    print(results)
    result_list+=results
    

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print("--------------------\n")
print(result_list)


file_name = f"results_{formatted_datetime}.json"


with open(os.path.join(base_path, file_name), 'w') as json_file:
    json.dump(result_list, json_file)


