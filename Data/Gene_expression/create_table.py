import os
import pandas as pd
import json


class JsonTxtDataProcessor:
    def __init__(self, json_file_path, txt_directory_path, num_files=None, uq=None):
        self.case_ids = None
        self.json_file_path = json_file_path
        self.txt_directory_path = txt_directory_path
        self.num_files = num_files
        self.uq = uq
        self.json_data = self.load_json()
        self.file_case_mapping = self.create_file_case_mapping()
        self.df = self.load_txt()

    def load_json(self):
        with open(self.json_file_path, 'r') as json_file:
            return json.load(json_file)

    def create_file_case_mapping(self):
        file_case_mapping = {}

        for entry in self.json_data:
            for case_info in entry['cases']:
                file_name = entry['file_name']
                file_case_mapping[file_name] = case_info['case_id']
        print(file_case_mapping)
        return file_case_mapping

    def load_txt(self):
        all_dfs = []
        self.case_ids = []

        for root, dirs, files in os.walk(self.txt_directory_path):
            for file in files:
                if file.endswith("counts.tsv"):
                    file_path = os.path.join(root, file)

                    column_name = self.file_case_mapping[file]
                    # print(column_name)

                    # header equal to 5 to skip the first header lines

                    if self.uq:
                        df = pd.read_csv(file_path, delimiter='\t', header=5,
                                     names=['gene_id', 'gene_name', 'gene_type', 'unstranded', 'stranded_first',
                                     'stranded_second', 'tpm_unstranded', 'fpkm_unstranded', column_name])
                    else:
                        df = pd.read_csv(file_path, delimiter='\t', header=5,
                                         names=['gene_id', 'gene_name', 'gene_type', 'unstranded', 'stranded_first',
                                                'stranded_second', 'tpm_unstranded', column_name, 'fpkm_uq_unstranded'])

                    #print(df)
                    #print(file)

                    df = df[['gene_id', column_name]]
                    df = df.set_index('gene_id')

                    #df = df[['gene_name', column_name]]
                    #df = df.set_index('gene_name')

                    #print(df)
                    all_dfs.append(df)

                    if self.num_files and len(all_dfs) >= self.num_files:
                        break

            if self.num_files and len(all_dfs) >= self.num_files:
                break

        if all_dfs:
            #print(all_dfs)
            return pd.concat(all_dfs, ignore_index=False, axis=1)
        else:
            raise ValueError("No files ending with 'counts.tsv' found in the specified directory.")

    def process_data(self):
        #print(self.df)
        return self.df


# Example usage:
json_file_path = 'gene_expression.json'  # Replace with actual json file
txt_directory_path = 'path/to/your/mRNA_data\\mRNA_data'  # Replace with the actual directory path
num_files_to_process = None  # Set to None to process all files

# https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/#fpkm
uq = None   # Set to None if uq is not wanted

data_processor = JsonTxtDataProcessor(json_file_path, txt_directory_path, num_files=num_files_to_process, uq=uq)
result_df = data_processor.process_data()
print(result_df)

# Save result_df as CSV file
if uq:
    result_df.to_csv('fpkm_uq.csv', index=True)
    pass
else:
    result_df.to_csv('fpkm.csv', index=True)
    pass
