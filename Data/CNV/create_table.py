import os
import pandas as pd
import json
import configparser



class JsonTxtDataProcessor:
    def __init__(self, json_file_path, txt_directory_path, num_files=None):
        self.case_ids = None
        self.json_file_path = json_file_path
        self.txt_directory_path = txt_directory_path
        self.num_files = num_files
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
                if file.endswith("number.v36.tsv"):
                    file_path = os.path.join(root, file)

                    column_name = self.file_case_mapping[file]
                    # print(column_name)

                    df = pd.read_csv(file_path, delimiter='\t', header=None,
                                     names=['gene_id', 'gene_name', 'chromosome', 'start', 'end', column_name,
                                            'min_copy_number', 'max_copy_number'])
                    #print(df)
                    #print(file)

                    df = df[['gene_name', column_name]]
                    df = df.set_index('gene_name')
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
            raise ValueError("No files found in the specified directory.")

    def process_data(self):
        #print(self.df)
        return self.df


# Example usage:    
parser = configparser.ConfigParser()
parser.read('C:\\Users\\pavon\\Documents\\Bioinformatics_project\\Data\\config.ini')
json_file_path = parser['cnv']['json_file']
txt_directory_path = parser['cnv']['path_data']
num_files_to_process = None  # Set to None to process all files

data_processor = JsonTxtDataProcessor(json_file_path, txt_directory_path, num_files=num_files_to_process)
result_df = data_processor.process_data()
print(result_df)

# Save result_df as CSV file
result_df.to_csv('result.csv', index=True)
