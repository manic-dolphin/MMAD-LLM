from llama2.llama import Llama
from planner import Planner
from models.models import Planner_Model
from graph import Thought, GoT, Operations

if __name__ == '__main__':
    pm = Planner_Model('llama2/llama-2-7b-chat/',
                       'llama2/tokenizer.model',
                       max_seq_len=4096
                       )
    # model = Llama.build(
    #         ckpt_dir='llama2/llama-2-13b-chat/',
    #         tokenizer_path='llama2/tokenizer.model',
    #         max_seq_len=512,
    #         max_batch_size=4
    #     )
    model = pm.get_model()
    planner_model = Planner(model=model,
                            max_gen_len=1024
                            )
    print(planner_model.get_initial_state("please prove that: \\lim_{n to \\infty} \\frac{n^{2} +1 }{2 * n^{2} - 7 * n} = 1/2."))
    initial_state = planner_model.get_initial_state("please prove that: \\lim_{n to \\infty} \\frac{n^{2} +1 }{2 * n^{2} - 7 * n} = 1/2.")
    # initial_state = planner_model.get_initial_state("Here is a chemical reaction formula: Reactants are:aryl halide:CCOC(=O)C1=CC2=C(O1)C(=CC=C2)Br;amine:C1CN(CCN1)CCC2=CC=CC=N2, Reagents are:Base:C(=O)([O-])[O-].[Cs+].[Cs+];Solvent:C1COCCO1;metal and ligand:CC(C)C1=CC(=C(C(=C1)C(C)C)C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C)C;metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd], and Products are 0:CCOC(=O)C1=CC2=C(O1)C(=CC=C2)N3CCN(CC3)CCC4=CC=CC=N4, please give me the reaction condition of this formula.")
    operations = Operations(model,
                            "please prove that: \\lim_{n to \\infty} \\frac{n^{2} +1 }{2 * n^{2} - 7 * n} = 1/2.",
                            max_gen_len=1024)
    initial_thought = Thought(initial_state)
    got = GoT(initial_thought=initial_thought,
              operations=operations,
              max_breadth=3,
              aggregate_prob=0.80
              )
    print(operations.generate(initial_thought).state)
    print("------------------")
    # print(operations.score(initial_thought))
    got.reason(7, 3)
    print(got.previous_thoughts)
    graph = got.consturct_graph(got.previous_thoughts)
    for node in graph:
        print('############################################################')
        print(node.state)
        print('############################################################')
    print(graph)