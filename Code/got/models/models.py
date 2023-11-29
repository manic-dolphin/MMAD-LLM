from llama2.llama import Llama

class Planner_Model:
    """
    """
    def __init__(self, ckpt_dir, tokenizer_dir, max_seq_len=512, max_batch_size=4):
        
        self.ckpt_dir: str = ckpt_dir
        self.tokenizer_dir: str = tokenizer_dir
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
    
    def get_model(self):
        
        model = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_dir,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )
        
        return model