import requests
import json
import re
import os
files_endpt = "https://api.gdc.cancer.gov/files"
with open('Data/query.json') as json_file:  
    filters = json.load(json_file)
resume = True
params = {}
if not resume:
    # Here a GET is used, so the filter parameters should be passed as a JSON string.
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id",
        "format": "JSON",
        "size": "1000"
        }

    response = requests.get(files_endpt, params = params)

    file_uuid_list = []
    batch_file_names = []
    all_file_names = []
    # This step populates the download list with the file_ids from the previous query
    for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
        file_uuid_list.append(file_entry["file_id"])

    data_endpt = "https://api.gdc.cancer.gov/data"
    data_dir = 'wsis/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    params = {"ids": file_uuid_list, "iter":0}
    with open("Data/ids.json", "w") as output_file:
        json.dump(params, output_file)
else:
    with open("Data/ids.json", "r") as input_file:
        params = json.load(input_file)
ids = params['ids']
start = params['iter']
batch_size = 10
for i, id in enumerate(ids):
    if i < start:
        continue
    print(f'starting from file {i}')
    params['iter'] = i
    with open("Data/ids.json", "w") as output_file:
        json.dump(params, output_file)
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(id)
    response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})

    # The file name can be found in the header within the Content-Disposition key.
    response_head_cd = response.headers["Content-Disposition"]

    file_name = os.path.join(data_dir, re.findall("filename=(.+)", response_head_cd)[0])
    batch_file_names.append(file_name)
    with open(file_name, "wb") as output_file:
        output_file.write(response.content)
    print(f"file {i} downloaded")
    if i % batch_size == 0 and i != 0:
        #DO THE PROCESSSING AND SAVE THE EMBEDDINGS HERE
        #remove files in the file_names list
        for file in batch_file_names:
            os.remove(file)
        all_file_names.extend(batch_file_names)
        batch_file_names = []
        print(f"batch {i} done")
names = {}
names['names'] = all_file_names
with open("Data/names.json", "w") as output_file:
    json.dump(names, output_file)