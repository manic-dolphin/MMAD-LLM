from llama2.llama import Llama

def init_llama_model():
    
    model = Llama.build(
            ckpt_dir='llama2/llama-2-13b-chat/',
            tokenizer_path='llama2/tokenizer.model',
            max_seq_len=4096,
            max_batch_size=4
        )
    
    return model

def get_message_for_llama(prompt: str):
    
    return [[{"role": "user", "content": prompt}]]