import pandas as pd

# Open and read the saved CSV file
opened_df = pd.read_csv('cnv.csv')
opened_df = opened_df.drop(['Unnamed: 0'], axis=1)

print(opened_df.head(20))
# Supponiamo che opened_df sia il tuo DataFrame
opened_df = opened_df.loc[(opened_df != 0).all(axis=1)]

print(opened_df.head(20))
print(opened_df.shape)

opened_df = opened_df.drop_duplicates(keep=False)

print(opened_df.head(20))
print(opened_df.shape)


#opened_df.to_csv('cnv_nan_removed.csv', index=True)