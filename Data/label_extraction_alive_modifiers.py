import json
import pandas as pd
import configparser
import matplotlib.pyplot as plt

def get_label_from_deads(file_path):
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
    

def get_label_from_the_alives(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        labels = {}
        for i in data:
            for j in i['diagnoses']:
                if 'days_to_last_follow_up' in j.keys():
                    labels[i['case_id']] = j['days_to_last_follow_up']
        for i in labels.keys():
            if labels[i] == None:
                labels[i] = -1
        return labels


def plot_labels(labels):
    
    plt.hist(labels.values(),bins=100)
    plt.show()

def alive_preprocessing(alive):
    #remove the -1 values and the ones < 1460
    flag = True
    while flag == True:
        flag = False
        for i in alive.keys():
            if alive[i] == -1 or alive[i] < 1825:
                del alive[i]
                flag = True
                break
    #increase the values by 25%
    for i in alive.keys():
        alive[i] = alive[i] * 1.25
    return alive
def age_extraction(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        ages = {}
        for i in data:
            if 'days_to_birth' in i['demographic'].keys():
                ages[i['case_id']] = -i['demographic']['days_to_birth']
        return ages
    

def filter(labels, th_low = 3*365, th_high = 5*365):
    filtered = {}
    flag = True
    for i in labels.keys():
        if labels[i] < th_low  or labels[i] > th_high:
            filtered[i] = labels[i]
    return filtered

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read('Data/config.ini')
    json_file_path = parser['clinical']['json_file_path']
    alive_file_path = parser['clinical']['alive_file_path']
    deads = get_label_from_deads(json_file_path)
    alives = get_label_from_the_alives(alive_file_path)
    alives = alive_preprocessing(alives)
    ages_alive = age_extraction(alive_file_path)
    ages_dead = age_extraction(json_file_path)
    ages = {**ages_alive, **ages_dead}
    #merge the two dictionaries alives and labels in a new one called complete
    complete = {**alives, **deads}
    filtered = filter(complete)
    df = pd.DataFrame.from_dict(alives, orient='index', columns=["days_to_death"])
    # Save whatever you need
    df2 = pd.DataFrame.from_dict(deads, orient='index', columns=["days_to_death"])
    df3 = pd.DataFrame.from_dict(complete, orient='index', columns=["days_to_death"])
    df4 = pd.DataFrame.from_dict(ages, orient='index', columns=["age"])
    df5 = pd.DataFrame.from_dict(filtered, orient='index', columns=["days_to_death"])
    output = parser['clinical']['output']
    if not output: output = 'labels.csv'
    tsv = output.endswith('.tsv')
    suffix = output[-4:]
    #Choose the files you need
    df2.to_csv(output[:-3] + '_only_dead' + suffix, header=df2.columns, index_label='id', sep='\t' if tsv else ',')
    df3.to_csv(output[:-3] + '_complete' + suffix, header=df3.columns, index_label='id', sep='\t' if tsv else ',')
    df5.to_csv(output[:-3] + '_complete_filtered' + suffix, header=df5.columns, index_label='id', sep='\t' if tsv else ',')
    df4.to_csv(output[:-3] + '_all_ages_at_diagnoses' + suffix, header=df4.columns, index_label='id', sep='\t' if tsv else ',')