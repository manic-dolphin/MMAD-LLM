import openai
import torch
from prompts import *
from utils import config

openai.api_key = config.OAI_KEY
model = "text-davinci-003"
# MAX_TOKENS_PER_GENERATION = 1024
SAMPLING_TEMPERATURE = 0.0


def generate(model, prompt):
    # model = "text-davinci-003"
    completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0,
            stop=["Question:"]
        )
    return completion.choices[0].text

def query_a_chat_completion(model, messages):
    assert model in ["gpt-3.5-turbo", "gpt-4"]
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # max_tokens=MAX_TOKENS_PER_GENERATION,
        temperature=SAMPLING_TEMPERATURE
    )
    return completion.choices[0].message.content

def solve_with_theorems(model, messages):
    assert model in ["gpt-3.5-turbo", "gpt-4"]
    prompt = messages[0]['content']
    theorems = sample_final_theorem_set("text-davinci-003", prompt)
    prompt = "Refer to or utilize the following theorems \
        (Note: It is not necessary to use all of these theorems. \
            Please make reasonable choices based on the specific requirements of the problem): " + theorems + \
        "to prove the following question." + prompt
    messages[0]['content'] = prompt

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # max_tokens=MAX_TOKENS_PER_GENERATION,
        temperature=SAMPLING_TEMPERATURE
    )

    return completion.choices[0].message.content, completion

if __name__ == '__main__':
    # output = generate(model, "say this is a test!")
    print("************************")
    messages = [
        {"role": "user", "content": 
         "hello"
        }
    ]
    output = query_a_chat_completion("gpt-3.5-turbo", messages=messages)
    print(output)
    print("************************")
    # output2 = solve_with_theorems("gpt-3.5-turbo", messages=messages)[0]
    # print(output2)