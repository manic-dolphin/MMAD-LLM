from llama2.llama import Llama
import os
import pandas as pd
import logging

logging.basicConfig(filename="proof.log", 
                    filemode='a',
                    format='%(message)s',
                    level=logging.INFO)

def load_data(textbook):
    assert textbook in ['mathematical_analysis', 'peiliwen']
    if textbook == 'peiliwen':
        data = pd.read_json(os.path.join(os.getcwd(), 'data/data_peiliwen.json'))
        # with open("//test.json",'r') as load_f:
        #     load_dict = json.load(load_f)
    if textbook == 'mathematical_anaylsis':
        pass
    
    return data

def score_with_llm(textbook:str,
                   model:str,
                   start_index,
                   solving_number,
                   max_seq_length=2048,
                   max_batch_size=4
                   ):
    data = load_data(textbook)
    
    if model == 'llama':
        generator = Llama.build(
            ckpt_dir='llama2/llama-2-13b-chat/',
            tokenizer_path='llama2/tokenizer.model',
            max_seq_len=max_seq_length,
            max_batch_size=max_batch_size,
        )
        for i in range(start_index, start_index + solving_number):
            prompt = data.iloc[i]['prompt']
            problem = [
            [{
                "role": "user",
                "content": prompt
            }]
            ]
            logging.info(f"Problem: {prompt}")
            proof = generator.chat_completion(
                    dialogs=problem,
                    max_gen_len=max_seq_length,
                    temperature=0.0
                    )
            logging.info(proof[0]['generation']['content'])
            logging.info('################################################')
        
        logging.info("Complete!")

if __name__ == '__main__':
    # data = load_data('peiliwen')
    # print(data.iloc[36]['prompt'])
    score_with_llm('peiliwen','llama', 51, 1) 