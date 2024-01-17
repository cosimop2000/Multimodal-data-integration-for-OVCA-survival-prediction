
from src.network.customics import CustOMICS
from src.tools.prepare_dataset import prepare_dataset
from src.tools.utils import get_sub_omics_df
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch

omics_df = {'protein': pd.read_csv('toy_data/protein.txt', sep='\t', index_col=0, header=0).T,
            'gene_exp': pd.read_csv('toy_data/gene_exp.txt', sep='\t', index_col=0, header=0).T,
            'methyl': pd.read_csv('toy_data/methyl.txt', sep='\t', index_col=0, header=0).T
            }
clinical_df = pd.read_csv('toy_data/labels.txt', sep='\t', index_col=1, header=0)


clinical_df.head()


lt_samples = list(clinical_df.index)


samples_train, samples_test = train_test_split(lt_samples, test_size=0.2)
samples_train, samples_val = train_test_split(samples_train, test_size=0.2)


omics_train = get_sub_omics_df(omics_df, samples_train)
omics_val = get_sub_omics_df(omics_df, samples_val)
omics_test = get_sub_omics_df(omics_df, samples_test)


x_dim = [omics_df[omic_source].shape[1] for omic_source in omics_df.keys()]

#### Defining Hyperparameters

batch_size = 32
n_epochs = 10
device = torch.device('cpu')
label = 'cluster.id'
event = 'cluster.id'
surv_time = 'cluster.id'

task = 'classification'
sources = ['gene_exp', 'methyl', 'protein']

hidden_dim = [512, 256]
central_dim = [512, 256]
rep_dim = 128
latent_dim=128
num_classes = 5
dropout = 0.2
beta = 1
lambda_classif = 5
classifier_dim = [128, 64]
lambda_survival = 0
survival_dim = [64,32]

source_params = {}
central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim, 'norm': True, 'dropout': dropout, 'beta': beta}
classif_params = {'n_class': num_classes, 'lambda': lambda_classif, 'hidden_layers': classifier_dim, 'dropout': dropout}
surv_params = {'lambda': lambda_survival, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout': dropout}
for i, source in enumerate(sources):
    source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim, 'norm': True, 'dropout': 0.2}
train_params = {'switch': 5, 'lr': 1e-3}


model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                        surv_params=surv_params, train_params=train_params, device=device).to(device)
print('Number of Parameters: ', model.get_number_parameters())
model.fit(omics_train=omics_train, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
            omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=True)
metric = model.evaluate(omics_test=omics_test, clinical_df=clinical_df, label=label, event=event, surv_time=surv_time,
                task=task, batch_size=1024, plot_roc=False)
model.plot_loss()



model.get_latent_representation(omics_df)



