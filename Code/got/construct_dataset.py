import os
import json
import pandas as pd

from datasets import Dataset

def load_json_file(data_dir):
    with open(data_dir, 'r') as file:
        data = json.load(file)
        
    return data

def merge_json_files(directory):
    merged_data = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            # 读取 JSON 文件的内容
            with open(filepath, "r") as file:
                data = json.load(file)
                merged_data.append(data)  # 将内容合并到一个列表中

    # 将列表中的所有内容合并成一个大的 JSON 对象
    merged_json = json.dumps(merged_data, indent=4)

    # 将合并后的 JSON 对象写入一个新的文件中
    with open("merged_data.json", "w") as outfile:
        outfile.write(merged_json)

def construct_dataset_for_gnn():
    pass

if __name__ == '__main__':
    raw_dataset = Dataset.load_from_disk('./data/chem_data/orderly_train')
    # raw_dataset_test = raw_dataset[10000: 15000]
    raw_dataset_test = Dataset.from_dict(raw_dataset[10000: 15000])
    print(raw_dataset_test)
    raw_dataset_test.save_to_disk('./data/chem_data/raw_dataset_test')
    # print(raw_dataset_test)
    # gnn_data_test = load_json_file('./extract_data/data_10000.json')
    # # gnn_data_test = pd.read_json('./extract_data/data_10000.json')
    # # print(len(raw_dataset_test[0]))
    # print(len(gnn_data_test))
    # print(gnn_data_test)