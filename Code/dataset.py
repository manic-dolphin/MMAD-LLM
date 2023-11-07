import torch
from llama2.llama import Llama

def generate_dataset(source_data,
                     generate_number):
    
    generator = Llama.build(
        ckpt_dir='llama2/llama-2-7b-chat/',
        tokenizer_path='llama2/tokenizer.model',
        max_seq_len=256,
        max_batch_size=4
        )
    result = []
    for i in range(len(source_data)):
        prompt = f"Given the original solution steps [{source_data[i]}], please rewrite and get one new solution step. The length of the new steps should not differ significantly from the original ones. Only the generated solution content needs to be returned. Since these are single-step solution steps, you only need to generate one step as well."
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
            ]]
        for j in range(generate_number):
            m = torch.distributions.Uniform(0.0, 1.5)
            temperature = m.sample()
            new_stpe = generator.chat_completion(
                    dialogs=messages,
                    temperature=float(temperature),
                    )
            result.append(new_stpe[0]['generation']['content'])
    
    return result
    
if __name__ == '__main__':
    example = """Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\)."""
    data = [example]
    print(generate_dataset(data, 4))