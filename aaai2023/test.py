import pandas as pd
import json
import torch

if __name__ == '__main__':
    # file = json.load("./aaai2023/data/data_mathematical_analysis.json")
    # print(len(file))
    # data = pd.read_json("./aaai2023/data/data_mathematical_analysis.json")
    # print(len(data))
    x = torch.linspace(0, 2, 5)
    print(x)
    print(str([' Mean Value Theorem ', ' Rolles Theorem ', ' Taylors Theorem ', ' Lagranges Mean Value Theorem']))
    messages = [
        {"role": "user", "content": 
         "please prove that: suppose the function f(x) is twice differentiable on the interval [a, b], f((a + b) / 2) = 0, let $M = \sup_{a \leq x \leq b} |f^{''}(x)|$, so $\int_{a}^{b} f(x){\rm d}x \leq M * (b - a)^3 / 24$."
        }
    ]
    print(messages[0]['content'])