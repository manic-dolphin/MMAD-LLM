from utils import config
import openai

openai.api_key = config.OAI_KEY

def divide_process(problem: str,
                   model: str,
                   prompt="Let's proof this problem step by step.",
                   temperature=0.0,
                   ):
    assert model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]

    messages = [
        {"role": "user", 
         "content": prompt + problem
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # max_tokens=MAX_TOKENS_PER_GENERATION,
        temperature=temperature
    )

    proof_process = response.choices[0].message.content
    proof_process = proof_process.split("Step")
    for i in range(len(proof_process)):
        proof_process[i] = proof_process[i].replace("\n", "")

    return proof_process

class GA():
    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    # model = "gpt-3.5-turbo"
    # model = "gpt-4"
    model = "gpt-3.5-turbo"
    # problem = "given that the functions $f(x)$ and $g(x)$ are continuous. Prove that $\\phi(x)=\\min\\{f(x),g(x)\\}$,$\\psi(x)=\\{f(x),g(x)\\}$ are also continuous."
    problem = "10 + 1 = ?"
    # problem = "please prove that: \\lim_{x to \\infty} \\sqrt[n]{n} = 1."
    print("-----------------------------------------")
    divided_response = divide_process(problem, model)
    print(divided_response)
