from utils import config
import openai
import torch
import time

openai.api_key = config.OAI_KEY

def divide_process(problem: str,
                   model: str,
                   prompt="Generate one step proof begin with \"Step 1\"",
                   temperature=0.1,
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
        # max_tokens=768,
        request_timeout=1200,
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

    def initialize_population(self, 
                              one_step_prompt,
                              model,
                              n: int,
                              prompt="Make a varient of the following prompt"
                              ):
        # tempratures = torch.linspace()
        # assert model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
        m = torch.distributions.Uniform(0.0, 2.0)
        temperatures = torch.tensor([m.sample() for i in range(n)])
        print(temperatures)
        populattions = []
        messages = [
            {
                "role": "user",
                "content": prompt + one_step_prompt
            }
        ]
        for i in range(len(temperatures)):
                new_prompt = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=128,
                    temperature=float(temperatures[i])
                    )
                print(new_prompt)
                populattions.append(new_prompt)
                time.sleep(30)

        return populattions

if __name__ == '__main__':
    # model = "gpt-3.5-turbo"
    # model = "gpt-4"
    model = "gpt-3.5-turbo"
    # problem = "given that the functions $f(x)$ and $g(x)$ are continuous. Prove that $\\phi(x)=\\min\\{f(x),g(x)\\}$,$\\psi(x)=\\{f(x),g(x)\\}$ are also continuous."
    # problem = "10 + 1 = ?"
    problem = "please prove that: \\lim_{x to \\infty} \\sqrt[n]{n} = 1."
    print("-----------------------------------------")
    divided_response = divide_process(problem, model)
    print(divided_response)
    ga = GA()
    print(ga.initialize_population("Rewrite the expression as \\lim_{x \\to \\infty} n^{\\frac{1}{n}} = 1.", "gpt-3.5-turbo", 10))
    example = "Certainly, let's prove that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\).\n \
        **Step 1:** Definition of the Limit\n \
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\).\n \
        **Step 2:** Take the Natural Logarithm \n \
        Let's consider the natural logarithm of both sides: \(\ln\left(\sqrt[n]{n}\right) = \frac{1}{n} \ln(n)\).\n \
        **Step 3:** Use L'Hôpital's Rule \n \
        We can now apply L'Hôpital's Rule to evaluate the limit: \n \
        \[ \n \
        \lim_{n \to \infty} \frac{1}{n} \ln(n). \n \
        \] \n \
        **Step 4:** Evaluate the Limit \n \
        The limit \(\lim_{n \to \infty} \frac{1}{n} \ln(n)\) is of the form \(\frac{\infty}{\infty}\), and we can apply L'Hôpital's Rule to find that it equals 0. \n \
        **Step 5:** Revert to the Original Limit \n \
        Since \(\lim_{n \to \infty} \frac{1}{n} \ln(n) = 0\), we can conclude that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). \n \
        **Step 6:** Conclusion \n \
        Therefore, we have successfully proven that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). This completes the proof."