from utils import config, tools
import openai
import torch
import time
import tqdm
import re
import logging
from llama2.llama import Llama
from typing import Optional

logging.basicConfig(filename="test.log", 
                    filemode='a',
                    format='%(message)s',
                    level=logging.INFO)

def divide_process(problem: str,
                   model: str,
                   prompt="Generate one step proof begin with \"Step 1\"",
                   temperature=0.0,
                   ):
    assert model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "llama"]

    messages = [
        {"role": "user", 
         "content": prompt + problem
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # max_tokens=768,
        # request_timeout=1200,
        temperature=temperature
    )

    proof_process = response.choices[0].message.content
    proof_process = proof_process.split("Step")
    for i in range(len(proof_process)):
        proof_process[i] = proof_process[i].replace("\n", "")

    return proof_process

class GA_LLAMA():
    def __init__(self, 
                 population_numbers: int, 
                 evolution_steps: int, 
                 max_length: int,
                 ckpt_dir: str,
                 tokenizer_path: str,
                 max_seq_len: int,
                 max_batch_size: int,
                 example= None,
                 model_parallel_size: Optional[int] = None,
                 top_p: float = 0.9,
                 initialize_max_temperature: float = 1.0,
                 mutation_prob: float = 0.9
                 ):
        self.population_numbers = population_numbers
        self.evolution_steps = evolution_steps
        self.max_length = max_length
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.top_p = top_p
        self.initialize_max_temperature = initialize_max_temperature
        self.mutation_prob = mutation_prob
        self.example = example

    def initialize_population(self, 
                              one_step_prompt,
                              prompt="Make a varient of the following prompt, the difference between the newly generated prompts and the original prompts should not be too significant."
                              ) -> list:
        # tempratures = torch.linspace()
        # assert model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
        prompt="""Make a varient of the following prompt, 
        the difference between the newly generated prompts and the original prompts should not be too significant."""
        prompt += f" and the new prompt's length can not exceed to the original prompt too much."
        m = torch.distributions.Uniform(0.0, self.initialize_max_temperature)
        n = self.population_numbers
        temperatures = torch.tensor([m.sample() for i in range(n)])
        populattions = []
        messages = [
            [{
                "role": "user",
                "content": prompt + one_step_prompt
            }]
        ]
        # print(one_step_prompt)
        # l = int(3.5 * max(len(one_step_prompt.split(" "))))
        for i in range(n):
                new_solution = self.generator.chat_completion(
                    dialogs=messages,
                    max_gen_len=36,
                    temperature=float(temperatures[i]),
                    top_p=self.top_p
                    )
                populattions.append(new_solution[0]['generation']['content'])

        return populattions
    
    def compute_fittness_score(self,
                               populations:list,
                               problem:str) -> list:
        
        population_numbers = self.population_numbers
        prompt = """Give a thorough consideration to this proof problem and, 
         in accordance with the rules of genetic algorithms, assign scores to each prompt in the following populations' list. 
         Each prompt represents a step in the proof process, and the scores should range from 0 to 1.
         Just provide a rating and return it in the form of a string, for example: "0.4, 0.8, 0.9,...,0.4".
         Here is the proof problem: """ + problem + " . and here is the populations list: " + str(populations) + f" We have {population_numbers} populations, so you need to return {population_numbers} scores in total."

        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]

        fittness_scores = self.generator.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=32
                    )

        fittness_scores = fittness_scores[0]['generation']['content']
        fittness_scores = re.findall(r"\d+\.?\d*", fittness_scores)
        # TODO
        while len(fittness_scores) != population_numbers:
                fittness_scores = self.generator.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=32
                    )
                fittness_scores = fittness_scores[0]['generation']['content']
                fittness_scores = re.findall(r"\d+\.?\d*", fittness_scores)

        return torch.softmax(torch.tensor([float(score) for score in fittness_scores]), dim=0)

    def crossover(self,
                  problem:str,
                  previous_steps:list,
                  parent_1:str,
                  parent_2:str,
                  ):
        
        crossover_prompt = """Given the original proof question and the past proof steps that have been generated, 
                in accordance with the rules of the genetic algorithm, 
                perform a crossover operation on the following two parent proof steps to generate a new step 
                that is one step closer to being correct and more reasonable. Just return the newly generated proof step."""
        
        parent_prompt = "Two parent steps are: " + parent_1 + " and " + parent_2
        
        prompt = "Here is the original proof problem: " + problem + "and the past proof steps: " + str(previous_steps) + ". " + crossover_prompt + parent_prompt
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        # l = int(3.5 * max(len(parent_1.split(' ')), len(parent_2.split(" "))))
        new_step = self.generator.chat_completion(
                    dialogs=messages,
                    temperature=0.0,
                    max_gen_len=48
                    )
        
        return new_step[0]['generation']['content']

    def mutation(self,
                 problem:str,
                 previous_steps:list,
                 parent:str,
                 ):
        
        mutation_prompt = """ "Considering the original proof question along with the previously generated proof steps, 
        following the rules of genetic algorithms, 
        apply a mutation operation to the following parent proof step to generate 
        a new step that is one step closer to being correct and more reasonable. 
        This is equivalent to not being confined to specific problem-solving strategies, 
        but rather exploring based on the original problem and the provided proof steps.
        Just return the newly generated proof step."""

        parent_prompt = "The parent step that requires mutation is: " + parent

        prompt = "Here is the original proof problem: " + problem + "and the past proof steps: " + str(previous_steps) + ". " + mutation_prompt + parent_prompt
        messages = [[
            {
                "role": "user",
                "content": prompt
            }
        ]]
        m = torch.distributions.Uniform(1.0, 1.5)
        temperature = m.sample()
        # l = int(3.5 * len(parent.split(' ')))
        new_step = self.generator.chat_completion(
                    dialogs=messages,
                    temperature=float(temperature),
                    max_gen_len=48
                    )
        
        return new_step[0]['generation']['content']
    
    def evolution(self,
                  problem:str,
                  debug=False):
        if debug == True:
            divided_process = self.example
        else:
            divided_process = divide_process(problem, "gpt-3.5-turbo")
        steps = len(divided_process)
        assert steps > 0
        previsou_steps = []
        population_numbers = self.population_numbers
        evolution_steps = self.evolution_steps
        start_time = time.time()
        logging.info("##################################")
        logging.info(f"Begin to generate new solution...")

        for i in range(steps):
            # initialize the population for each step
            populations = self.initialize_population(divided_process[i])
            logging.info(f"No {i + 1} proof step.")
            # evolution
            for j in range(evolution_steps):

                logging.info(f"evolution step:{j + 1}")
                logging.info(f"current population: {populations}")
                
                # compute the fittness scores
                fittness_scores = self.compute_fittness_score(
                                                            populations,
                                                            problem
                                                            )
                logging.info(f"fittness scores: {fittness_scores}")
                
                # roulette wheel selection rule 
                probs = fittness_scores
                samples = torch.multinomial(probs, population_numbers, replacement=True)
                logging.info(f"sample index: {samples}")
                # BUG
                assert len(samples) == len(populations) == len(probs)
                selected_populations = [populations[i] for i in samples]

                new_populations = []
                for k in range(population_numbers):
                    # sample two parents
                    sampled_indices = torch.randint(low=0, high=population_numbers, size=(2,))
                    parent_1 = selected_populations[sampled_indices[0]]
                    parent_2 = selected_populations[sampled_indices[1]]

                    # crossover process
                    new_step = self.crossover(problem,
                                              previsou_steps,
                                              parent_1=parent_1,
                                              parent_2=parent_2
                                              )
                    # mutation process
                    random_number = torch.rand(1)
                    if random_number <= self.mutation_prob:
                        new_step = self.mutation(problem,
                                                previsou_steps,
                                                parent=new_step
                                                )
                    new_populations.append(new_step)
                populations = new_populations

            # recompute the fittness scores of the generated populations
            final_fittness_scores = self.compute_fittness_score(
                                                                populations,
                                                                problem
                                                                )
            
            # append the optimal solution to the previous steps
            optimal_step = populations[torch.argmax(final_fittness_scores)]
            previsou_steps.append(optimal_step)

            logging.info(f"optimal solution for current step: {optimal_step}")
            logging.info("--------------------------------------------------------------")

        logging.info(f"Optimized solutions: {previsou_steps}")
        logging.info("Complete!")
        end_time = time.time()
        hour, minute, second = tools.time_counter(start_time, end_time)
        logging.info(f"Solving process takes {hour}h{minute}min{second}sec.")

        return previsou_steps

if __name__ == '__main__':
    # model = "gpt-3.5-turbo"
    # model = "gpt-4"
    model = "gpt-3.5-turbo"
    # problem = "given that the functions $f(x)$ and $g(x)$ are continuous. Prove that $\\phi(x)=\\min\\{f(x),g(x)\\}$,$\\psi(x)=\\{f(x),g(x)\\}$ are also continuous."
    # problem = "10 + 1 = ?"
    problem = "please prove that: \\lim_{x to \\infty} \\sqrt[n]{n} = 1."
    # divided_response = divide_process(problem, model)
    # ['', ' 1: Rewrite the expression as a limit statement: \\lim_{n \\to \\infty} \\sqrt[n]{n} = 1.This step is necessary to clearly indicate that we are taking the limit as n approaches infinity.']
    # print(divided_response)
    ga_llama = GA_LLAMA(
        population_numbers=3,
        evolution_steps=3,
        max_length=1024,
        ckpt_dir='llama2/llama-2-7b-chat/',
        tokenizer_path='llama2/tokenizer.model',
        max_seq_len=2048,
        max_batch_size=4
    )
    # print(ga.initialize_population("Rewrite the expression as \\lim_{x \\to \\infty} n^{\\frac{1}{n}} = 1.", "gpt-3.5-turbo", 5))
    EXAMPLE = ["Certainly, let's prove that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\).",
        """Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\).""",
        """Step 2: Take the Natural Logarithm
        Let's consider the natural logarithm of both sides: \(\ln\left(\sqrt[n]{n}\right) = \frac{1}{n} \ln(n)\).""",
        """Step 3: Use L'Hôpital's Rule
        We can now apply L'Hôpital's Rule to evaluate the limit:
        \[
        \lim_{n \to \infty} \frac{1}{n} \ln(n). 
        \] """,
        """Step 4: Evaluate the Limit
        The limit \(\lim_{n \to \infty} \frac{1}{n} \ln(n)\) is of the form \(\frac{\infty}{\infty}\), 
        and we can apply L'Hôpital's Rule to find that it equals 0. """,
        """Step 5: Revert to the Original Limit
        Since \(\lim_{n \to \infty} \frac{1}{n} \ln(n) = 0\), we can conclude that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). """,
        """Step 6: Conclusion
        Therefore, we have successfully proven that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). This completes the proof."""]
    example_populations = ['Rewrite the expression as \\(\\lim_{n \\to \\infty} n^{\\frac{1}{n}} = 1\\).', 
                           'Redefine the limit of the expression as x approaches infinity in the following manner:\n\n\\lim_{x \\to \\infty} \\left(\\frac{x}{x + 1}\\right)^{\\frac{x + 1}{x}} = 1.', 
                           'Rewrite the limit expression as \\(\\lim_{n \\to \\infty} n^{\\frac{1}{n}} = 1\\)', 
                           'Prove that as \\(n\\) tends to infinity, the sequence \\(a_n = nth\\) converges to \\(L\\) where \\(L\\) is given by \\(\\lim_{n \\to \\infty} (1 + h)^{\\frac{1}{h}}\\) for every \\(h > 0\\).', 
                           'Prove that as $n$ approaches infinity, the sequence $\\sqrt[n]{n}$ has a limit equal to 1.']
    # print(ga.compute_fittness_score("gpt-3.5-turbo", example_populations, problem))
    # print(example_populations[1])
    # print(example_populations[2])
    print("---------------------------------------------")
    print(ga_llama.initialize_population(EXAMPLE[0]))
    print("---------------------------------------------")