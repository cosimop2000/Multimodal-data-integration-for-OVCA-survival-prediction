import pandas as pd

# Open and read the saved CSV file
df = pd.read_csv('fpkm.csv')
df
df.set_index('gene_id')

print(df.shape)
print(df.columns)