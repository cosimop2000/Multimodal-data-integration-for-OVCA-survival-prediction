import json
import pandas as pd
import configparser


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
            if 'days_to_death' in i['demographic'].keys():
                labels[i['case_id']] = i['demographic']['days_to_death']
        return labels

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read('Data/config.ini')
    json_file_path = parser['clinical']['json_file_path']
    labels = get_label(json_file_path)
    print(len(labels))
    df = pd.DataFrame.from_dict(labels, orient='index')
    df.to_csv('labels.csv', header=False)
    