import torch
from llama2.llama import Llama

def generate_dataset(source_data,
                     generate_number,
                     use_templete=True):
    
    generator = Llama.build(
        ckpt_dir='llama2/llama-2-7b-chat/',
        tokenizer_path='llama2/tokenizer.model',
        max_seq_len=512,
        max_batch_size=4
        )
    result = []
    # TODO
    templete = """
        Here is some examples:
        Examlple 1:
        Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\). ->
        Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon <= 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|n - 1\right| < \epsilon\).
        Example 2:
        Step 3: Use L'Hôpital's Rule
        We can now apply L'Hôpital's Rule to evaluate the limit:
        \[
        \lim_{n \to \infty} \frac{1}{n} \ln(n). ->
        Step 3:
        We can now apply L'Hôpital's Rule to evaluate the limit:
        \[
        \lim_{n \to \infty} n \ln(n). 
        \]
        \]
        """
        
    for i in range(len(source_data)):
        # prompt = f"Given the original solution steps [{source_data[i]}], please rewrite and get one new solution step. The length of the new steps should not differ significantly from the original ones. Only the generated solution content needs to be returned. Since these are single-step solution steps, you only need to generate one step as well."
        prompt = f"Now, given the original solution step: [{source_data[i]}], rewrite this step to generate a bad one, the length of the generated step should not differ significantly from the original one: ->"
        if use_templete:
            prompt = "Please follow this templete: " + templete + prompt
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
            ]]
        for j in range(generate_number):
            m = torch.distributions.Uniform(0.0, 1.0)
            temperature = m.sample()
            new_stpe = generator.chat_completion(
                    dialogs=messages,
                    temperature=float(temperature),
                    max_gen_len=128
                    )
            result.append(new_stpe[0]['generation']['content'])
    
    return result
    
if __name__ == '__main__':
    example = """Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\)."""
    data = [example]
    print(generate_dataset(data, 1, use_templete=False))