import argparse
from utils import *
from llama2 import *
from data import *
from evolution_llama import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    problem = "please prove that: \(\lim_{n \to \infty} \sqrt[n]{n} = 1\)."
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
    ga_llama = GA_LLAMA(
        population_numbers=3,
        evolution_steps=3,
        max_length=1024,
        ckpt_dir='llama2/llama-2-7b-chat/',
        tokenizer_path='llama2/tokenizer.model',
        max_seq_len=2048,
        max_batch_size=4,
        example=EXAMPLE
    )
    ga_llama.evolution(problem, debug=True)