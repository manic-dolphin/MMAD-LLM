from datasets import Dataset
from evaluation import hf_orderly_dataset
import pandas as pd
from tqdm import tqdm
import json

if __name__ == '__main__':
    # train_dataset, test_dataset = hf_orderly_dataset()
    # raw_train_data = pd.read_parquet("./data/chem_data/orderly_condition_train.parquet")
    # print(raw_train_data.columns)
    # count = 0
    # for i in tqdm(range(len(train_dataset))):
    #     example = train_dataset[i]
    #     if example['temperature'] == None:
    #         count += 1
    # no_temperature_rate = count / len(train_dataset)
    # print(no_temperature_rate)
    graph_data = pd.read_csv('./extract_data_using_knowledge/data_sample.csv')
    print(graph_data.iloc[1200]['reactions'])
    data_raw = Dataset.load_from_disk("./data/chem_data/orderly_train")
    print(data_raw[1200])
    print(len(graph_data))
    print(len(data_raw))