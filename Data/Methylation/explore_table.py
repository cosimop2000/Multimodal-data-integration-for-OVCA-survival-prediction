import pandas as pd

# Open and read the saved CSV file
opened_df = pd.read_csv('result.csv')
opened_df
opened_df.set_index('cg_site')

print(opened_df.shape)
print(opened_df.columns)