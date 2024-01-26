import pandas as pd

# Open and read the saved CSV file
opened_df = pd.read_csv('cnv.csv')
#opened_df.set_index('gene_name')
opened_df = opened_df.drop(['Unnamed: 0'], axis=1)
print(opened_df.shape)
print(opened_df.columns)
print(opened_df.head(20))

#opened_df.fillna(value=0.0, inplace=True)
#opened_df.to_csv('cnv.csv', index=True)