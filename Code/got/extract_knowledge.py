import os
import time
import json
import openai
from tqdm import tqdm
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

from datasets import Dataset

from templates import *

MAX_API_RETRY = 15

if __name__ == '__main__':
  train_data = Dataset.load_from_disk("./data/chem_data/orderly_train")
  data = {}
  
  for i in tqdm(range(30000, 35000)):
    
    # example = train_data[i]
    
    for _ in range(MAX_API_RETRY):
      try:
        reaction = train_data[i]['reaction']
        content = PARSING.format(reaction)
        messages = [
          {"role": "system", 
         "content": SYSTEM_PROMPT
          },
          {"role": "user", 
          "content": content
          }
        ]
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        )
        output = response.choices[0].message.content
        data.update({i : output})
        # time.sleep(5)

      except Exception as e:
        print(e)
        data.update({i : 'Error'})
        time.sleep(1)
      else:
        break
    if i % 100 == 0:
        with open('data_30000.json', "w") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=2)

  with open('data_30000.json', 'w') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=2)