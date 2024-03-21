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

    # 从文件加载 JSON 数据
    with open('data.json', 'r') as f:
        data = json.load(f)

    # 将包含 \u 的字符串转换为正常字符
    def decode_unicode(data):
        if isinstance(data, dict):
            return {decode_unicode(key): decode_unicode(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [decode_unicode(item) for item in data]
        elif isinstance(data, str):
            return data.encode().decode('unicode_escape')
        else:
            return data

    decoded_data = decode_unicode(data)

    # 将转换后的数据保存到新的 JSON 文件中
    with open('decoded_data.json', 'w') as f:
        json.dump(decoded_data, f, indent=4)

    print("Data saved to decoded_data.json.")
