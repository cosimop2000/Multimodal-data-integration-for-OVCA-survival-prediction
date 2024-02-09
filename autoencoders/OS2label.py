import pandas as pd
import configparser, os

parser = configparser.ConfigParser()
parser.read('../Data/config.ini')
base_path = parser['autoencoder']['base_path']   

clinical = pd.read_csv(os.path.join(base_path, 'TCGA-OV.survival.tsv'), sep='\t', index_col=0, header=0)

clinical["OS.label"] = clinical["OS.time"].apply(lambda x: 1 if x >4*365 else 0)

print(clinical)

clinical.to_csv(os.path.join(base_path, 'TCGA-OV.survival_labeled.tsv'), sep='\t', index=True, header=True, index_label='sample')