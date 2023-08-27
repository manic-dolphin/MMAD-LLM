import openai
import re
import torch

oai_key = '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'
openai.api_key = oai_key
MAX_TOKENS_PER_GENERATION = 1024
SAMPLING_TEMPERATURE = 0.0

def generate(model, prompt, temperature):
    # model = "text-davinci-003"
    completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=MAX_TOKENS_PER_GENERATION,
            temperature=temperature,
            stop=["Question:"]
        )
    return completion.choices[0].text

def sample_one_theorem_set(model,
                          prompt,
                          temperature=SAMPLING_TEMPERATURE):
    
    theorem_prompt = "Please provide me with the names of the theorems that may be needed to prove this proof question. \
        Please note that you only need to provide the names of the theorems, not the proof process."
    
    prompt = theorem_prompt + prompt
    theorem_list = generate(model, prompt, temperature).split('.')
    outputs = []
    for theorem in theorem_list:
        output = re.sub(r'[^a-zA-Z\s]', '', theorem).replace('\n', '')
        if len(output) >= 1 and len(output) <= 45:
            outputs.append(output)

    return outputs

def sample_final_theorem_set(model,
                             prompt,
                             sample_steps=10):
    tempratures = torch.linspace(0, 2, sample_steps)
    theorems = []
    for temperature in tempratures:
        outputs = sample_one_theorem_set(model, prompt, float(temperature))
        theorems.append(outputs)

    theorems = [element for theorem in theorems for element in theorem]
    prompt = "Theorems: " + str(theorems) + "The elements in this list are some theorems. \
        Please summarize the theorems that appear most frequently and return them to me. \
            Note that you only need to provide the names of the theorems. Nothing else needs to be said."
    theorems = generate(model, prompt, temperature=0.0)

    return theorems

if __name__ == "__main__":
    prompt = "please prove that: suppose the function f(x) is twice differentiable on the interval [a, b], f((a + b) / 2) = 0, let $M = \\sup_{a \\leq x \\leq b} |f^{''}(x)|$, so $\\int_{a}^{b} f(x){\rm d}x \\leq M * (b - a)^3 / 24$."
    model = "text-davinci-003"
    outputs = sample_one_theorem_set(model, prompt)
    print(outputs)
    theorems = sample_final_theorem_set(model, prompt)
    print(theorems)