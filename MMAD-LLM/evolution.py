from utils import config
import openai
import torch
import time

openai.api_key = config.OAI_KEY

def divide_process(problem: str,
                   model: str,
                   prompt="Generate one step proof begin with \"Step 1\"",
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
                              ) -> list:
        # tempratures = torch.linspace()
        # assert model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
        m = torch.distributions.Uniform(0.0, 2.0)
        temperatures = torch.tensor([m.sample() for i in range(n)])
        # print(temperatures)
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
                # print(new_prompt)
                populattions.append(new_prompt.choices[0].message.content)
                time.sleep(30)

        return populattions
    
    def compute_fittness_score(self,
                               model,
                               populations:list,
                               problem:str) -> list:
        
        prompt = """Give a thorough consideration to this proof problem and, 
         in accordance with the rules of genetic algorithms, assign scores to each prompt in the following populations' list. 
         Each prompt represents a step in the proof process, and the scores should range from 0 to 1.
         Just provide a rating and return it in the form of a string, for example: "0.4, 0.8, 0.9,...,0.4".
         Here is the proof problem: """ + problem + " . and here is the popupations: " + str(populations)

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        fittness_scores = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=128,
                    temperature=0.0
                    )

        fittness_scores = fittness_scores.choices[0].message.content

        return torch.softmax(torch.tensor([float(score.strip()) for score in fittness_scores.split(",")]), dim=0)
    
    def crossover(self,
                  problem:str,
                  previous_steps:list,
                  parent_1:str,
                  parent_2:str,
                  model):
        
        crossover_prompt = """Given the original proof question and the past proof steps that have been generated, 
                in accordance with the rules of the genetic algorithm, 
                perform a crossover operation on the following two parent proof steps to generate a new step 
                that is one step closer to being correct and more reasonable. Just return the newly generated proof step."""
        
        parent_prompt = "Two parent steps are: " + parent_1 + " and " + parent_2
        
        prompt = "Here is the original proof problem: " + problem + "and the past proof steps: " + str(previous_steps) + ". " + crossover_prompt + parent_prompt
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        new_step = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=128,
                    temperature=0.0
                    )
        
        return new_step.choices[0].message.content

    def mutation(self,
                 problem:str,
                 previous_steps:list,
                 parent:str,
                 model):
        
        mutation_prompt = """ "Considering the original proof question along with the previously generated proof steps, 
        following the rules of genetic algorithms, 
        apply a mutation operation to the following parent proof step to generate 
        a new step that is one step closer to being correct and more reasonable. 
        This is equivalent to not being confined to specific problem-solving strategies, 
        but rather exploring based on the original problem and the provided proof steps.
        Just return the newly generated proof step."""

        parent_prompt = "The parent step that requires mutation is: " + parent

        prompt = "Here is the original proof problem: " + problem + "and the past proof steps: " + str(previous_steps) + ". " + mutation_prompt + parent_prompt
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        m = torch.distributions.Uniform(1.0, 2.0)
        temperature = m.sample()
        new_step = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=180,
                    temperature=float(temperature)
                    )
        
        return new_step.choices[0].message.content
    
    def evolution(self,
                  problem:str,
                  model="gpt-3.5-turbo",
                  population_numbers=5,
                  evolution_steps=5,
                #   crossover_prob=0.75,
                  mutation_prob=0.90,
                  debug=False):
        if debug == True:
            divided_process = EXAMPLE
        else:
            divided_process = divide_process(problem, "gpt-3.5-turbo")
        steps = len(divided_process)
        assert steps > 0
        previsou_steps = []

        for i in range(steps):
            # initialize the population for each step
            populations = self.initialize_population(divided_process[i],
                                                     model,
                                                     population_numbers)
            
            # evolution
            for j in range(evolution_steps):
                # compute the fittness scores
                fittness_scores = self.compute_fittness_score(model,
                                                            populations,
                                                            problem)
                
                # roulette wheel selection rule 
                probs = fittness_scores.copy()
                samples = torch.multinomial(probs, population_numbers, replacement=True)
                selected_populations = [populations[i] for i in samples]
                assert len(selected_populations) == population_numbers

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
                                              parent_2=parent_2,
                                              model=model)
                    # mutation process
                    random_number = torch.rand(1)
                    if random_number <= mutation_prob:
                        new_step = self.mutation(problem,
                                                previsou_steps,
                                                parent=new_step,
                                                model=model)
                    new_populations.append(new_step)
                populations = new_populations

            final_fittness_scores = self.compute_fittness_score(model,
                                                                populations,
                                                                problem)
            optimal_step = populations[torch.argmax(final_fittness_scores)]
            previsou_steps.append(optimal_step)

        return previsou_steps

if __name__ == '__main__':
    # model = "gpt-3.5-turbo"
    # model = "gpt-4"
    model = "gpt-3.5-turbo"
    # problem = "given that the functions $f(x)$ and $g(x)$ are continuous. Prove that $\\phi(x)=\\min\\{f(x),g(x)\\}$,$\\psi(x)=\\{f(x),g(x)\\}$ are also continuous."
    # problem = "10 + 1 = ?"
    problem = "please prove that: \\lim_{x to \\infty} \\sqrt[n]{n} = 1."
    print("-----------------------------------------")
    # divided_response = divide_process(problem, model)
    # ['', ' 1: Rewrite the expression as a limit statement: \\lim_{n \\to \\infty} \\sqrt[n]{n} = 1.This step is necessary to clearly indicate that we are taking the limit as n approaches infinity.']
    # print(divided_response)
    ga = GA()
    # print(ga.initialize_population("Rewrite the expression as \\lim_{x \\to \\infty} n^{\\frac{1}{n}} = 1.", "gpt-3.5-turbo", 5))
    EXAMPLE = ["Certainly, let's prove that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\).",
        """Step 1: Definition of the Limit
        We want to prove that for any \(\epsilon > 0\), there exists a positive integer \(N\) 
        such that for all \(n > N\), \(\left|\sqrt[n]{n} - 1\right| < \epsilon\).""",
        """Step 2: Take the Natural Logarithm
        Let's consider the natural logarithm of both sides: \(\ln\left(\sqrt[n]{n}\right) = \frac{1}{n} \ln(n)\).""",
        """Step 3:** Use L'Hôpital's Rule
        We can now apply L'Hôpital's Rule to evaluate the limit:
        \[
        \lim_{n \to \infty} \frac{1}{n} \ln(n). 
        \] """,
        """Step 4: Evaluate the Limit
        The limit \(\lim_{n \to \infty} \frac{1}{n} \ln(n)\) is of the form \(\frac{\infty}{\infty}\), 
        and we can apply L'Hôpital's Rule to find that it equals 0. """,
        """Step 5:** Revert to the Original Limit
        Since \(\lim_{n \to \infty} \frac{1}{n} \ln(n) = 0\), we can conclude that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). """,
        """Step 6: Conclusion
        Therefore, we have successfully proven that \(\lim_{n \to \infty} \sqrt[n]{n} = 1\). This completes the proof."""]
    example_populations = ['Rewrite the expression as \\(\\lim_{n \\to \\infty} n^{\\frac{1}{n}} = 1\\).', 
                           'Redefine the limit of the expression as x approaches infinity in the following manner:\n\n\\lim_{x \\to \\infty} \\left(\\frac{x}{x + 1}\\right)^{\\frac{x + 1}{x}} = 1.', 
                           'Rewrite the limit expression as \\(\\lim_{n \\to \\infty} n^{\\frac{1}{n}} = 1\\)', 
                           'Prove that as \\(n\\) tends to infinity, the sequence \\(a_n = nth\\) converges to \\(L\\) where \\(L\\) is given by \\(\\lim_{n \\to \\infty} (1 + h)^{\\frac{1}{h}}\\) for every \\(h > 0\\).', 
                           'Prove that as $n$ approaches infinity, the sequence $\\sqrt[n]{n}$ has a limit equal to 1.']
    print(ga.compute_fittness_score("gpt-3.5-turbo", example_populations, problem))
    print(example_populations[1])
    print(example_populations[2])
    print("---------------------------------------------")
    print(ga.evolution(problem, "gpt-3.5-turbo", debug=True))
    print("---------------------------------------------")
