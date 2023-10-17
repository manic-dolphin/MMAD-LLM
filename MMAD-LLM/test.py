import pandas as pd
import json
import torch

if __name__ == '__main__':
    # file = json.load("./aaai2023/data/data_mathematical_analysis.json")
    # print(len(file))
    # data = pd.read_json("./aaai2023/data/data_mathematical_analysis.json")
    # print(len(data))
    # x = torch.linspace(0, 2, 5)
    # print(x)
    # print(str([' Mean Value Theorem ', ' Rolles Theorem ', ' Taylors Theorem ', ' Lagranges Mean Value Theorem']))
    # messages = [
    #     {"role": "user", "content": 
    #      "please prove that: suppose the function f(x) is twice differentiable on the interval [a, b], f((a + b) / 2) = 0, let $M = \sup_{a \leq x \leq b} |f^{''}(x)|$, so $\int_{a}^{b} f(x){\rm d}x \leq M * (b - a)^3 / 24$."
    #     }
    # ]
    # print(messages[0]['content'])
    print(str([1, 2, 3]))
    print("[0.9, 0.7, 0.9, 0.4, 0.8]".split(","))
    import torch

    # 指定的分布
    probs = torch.tensor([0.1988, 0.2198, 0.1988, 0.1628, 0.2198])

    # 采样的次数
    num_samples = 10  # 你可以根据需要更改采样次数

    # 使用torch.multinomial()进行采样
    samples = torch.multinomial(probs, num_samples, replacement=True)

    # 输出采样结果
    print(samples)
    # 从0到9之间随机采样两个索引
    sampled_indices = torch.randint(low=0, high=10, size=(2,))

    # 输出采样结果
    print(sampled_indices[0])
    print(torch.rand(1))
