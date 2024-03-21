GENERAL_CONDITION_TEMPLATE = """
You are an expert chemist. Given the reaction formula, which includes reactants and products in the form of SMILES.
Your task is to predict the reaction condition of the reaction using your experienced chemical knowledge.
Reaction condition you need to predict consist of the agents and the solvents. 
"""

IN_CONTEXT_LEARNING_CONDITION_TEMPLETE = """
Now, please strictly follow the format, no other information can be provided!
You will be provided with several examples reactions, each accompanied by the corresponding reaction conditions.
Please predict the reaction conditions now.

Here is a chemical reaction. Reactants are: NCC(=O)N[C@H](CC1CCCCC1)C(=O)O, O=C(O)COc1ccc([C@@H]2[C@@H](SCC(=O)c3ccc4c(c3)CCO4)C(=O)N2c2ccc(F)cc2)cc1. Product is: O=C(COc1ccc([C@@H]2[C@@H](SCC(O)c3ccc4c(c3)CCO4)C(=O)N2c2ccc(F)cc2)cc1)NCC(=O)N[C@H](CC1CCCCC1)C(=O)O
The reaction conditions of this reaction are: Agents: F[B-](F)(F)F, CN(C)C(On1nnc2ccccc21)=[N+](C)C. Solvents: CN1CCOCC1, CN(C)C=O

Here is a chemical reaction. Reactants are: COC(=O)c1ccc(S(=O)(=O)C2CCOC2)cc1. Product is: O=C(O)c1ccc(S(=O)(=O)C2CCOC2)cc1
The reaction conditions of this reaction are: Agents: [Na+], [OH-]. Solvents: C1COCCO1

Here is a chemical reaction. Reactants are: CC(C)(C)NCC(=O)c1ccc(O)c(O)c1, O=C(Cl)c1ccccc1. Product is: CC(C)(C)NCC(=O)c1ccc(OC(=O)c2ccccc2)c(O)c1
The reaction conditions of this reaction are: Agents: [Na+], C[O-], Cl. Solvents: CN(C)C=O

Here is a chemical reaction. Reactants are: CCOC(=O)C(=NOC)c1csnn1. Product is: CON=C(C(=O)O)c1csnn1
The reaction conditions of this reaction are: Agents: [OH-], [Na+]. Solvents: CO, O

Here is a chemical reaction. Reactants are: Cc1cc2[nH]c(=O)n(C3=CCN(Cc4ccccc4)CC3)c2cc1C. Product is: Cc1cc2[nH]c(=O)n(C3=CCNCC3)c2cc1C
The reaction conditions of this reaction are: Agents: [H][H], [Pd]. Solvents: CC(=O)O, CCO

Here is a chemical reaction. Reactants are: CN(C)CCc1cn(C(=O)c2ccccc2)c2ccc(C3CCC=CO3)cc12. Product is: CN(C)CCc1c[nH]c2ccc(C3CCC=CO3)cc12
The reaction conditions of this reaction are: Agents: [OH-], [K+]. Solvents: CO

Here is a chemical reaction. Reactants are: COc1ccc(CNc2ncccc2-c2cn3c(CN4CCOCC4)csc3n2)cc1. Product is: Nc1ncccc1-c1cn2c(CN3CCOCC3)csc2n1
The reaction conditions of this reaction are: Agents: CC[SiH](CC)CC. Solvents: ClCCl, O=C(O)C(F)(F)F

Here is a chemical reaction. Reactants are: N#Cc1ccc(-c2ccc(F)cc2)nc1Cl, N#Cc1ccc(NC2CCCNC2)nc1N. Product is: N#Cc1ccc(NC2CCCN(c3nc(-c4ccc(F)cc4)ccc3C#N)C2)nc1N
The reaction conditions of this reaction are: Agents: Cl. Solvents: CCN(C(C)C)C(C)C, CS(C)=O

{}

"""


REACTION_CONDITION_COT_0 = """
    Generate a detailed functional group analysis based on the following specific chemical reaction equation. The chemical reaction is: {}.
    The analysis should include a thorough understanding of the types and positions of functional groups in the reactants and products. 
    Provide detailed information about the functional groups, covering their structures, chemical properties, and potential transformations during the reaction. 
    Ensure that your analysis reflects a rich understanding of chemical knowledge and use scientific and chemical terminology for detailed explanations.
    Please consider the following:
    1.Identify and describe the functional groups present in the reactants and products.
    2.Analyze potential changes in functional groups during the reaction, including possible additions, eliminations, or transformations.
    3.Describe the structures and properties of the functional groups, as well as their potential roles in the reaction.
    4.Consider any potential catalytic or participatory roles of functional groups in the reaction.
    5.Use professional scientific terminology to ensure that the generated analysis reflects a profound understanding of functional groups.
    Now, give me your response: 
"""

REACTION_CONDITION_COT_1 = """
    Based on the specific chemical reaction equation provided below, provide a detailed analysis to gain a profound understanding of the essence of the reaction. The chemical reaction is: {}.
    Focus on the reaction type, reaction conditions, and potential involvement of novel catalysts.
    Please elaborate on the following aspects, incorporating chemical knowledge for explanation:
    1.Reaction Type: Determine and elaborate on the specific type of the reaction, such as nucleophilic substitution, addition reaction, redox reaction, etc. Explain how the substrate structure and reaction mechanism influence the categorization.
    2.Reaction Conditions: Analyze the applicable reaction conditions, including temperature, pressure, solvent selection, etc. Provide detailed explanations of how these conditions impact the reaction rate and selectivity, considering principles from reaction kinetics and thermodynamics.
    3.Novel Catalysts: Consider whether novel catalysts are involved in the reaction. If so, describe their structure, potential catalytic mechanisms, and advantages compared to traditional catalysts, incorporating principles of catalyst design.
    4.Please use professional scientific and chemical terminology to ensure that the generated analysis fully reflects a profound understanding of the chemical reaction.
    Now, give me your response:
"""

REACTION_CONDITION_COT_2 = """
    Based on the specific chemical reaction equation provided below, provide a detailed analysis to gain a profound understanding of the essence of the reaction. The chemical reaction is: {}.
    Focus on the reaction type, reaction conditions, and potential involvement of novel catalysts.
    Please elaborate on the following aspects, incorporating chemical knowledge for explanation:
    1.Reaction Type: Determine and elaborate on the specific type of the reaction, such as nucleophilic substitution, addition reaction, redox reaction, etc. Explain how the substrate structure and reaction mechanism influence the categorization.
    2.Reaction Conditions: Analyze the applicable reaction conditions, including temperature, pressure, solvent selection, etc. Provide detailed explanations of how these conditions impact the reaction rate and selectivity, considering principles from reaction kinetics and thermodynamics.
    3.Novel Catalysts: Consider whether novel catalysts are involved in the reaction. If so, describe their structure, potential catalytic mechanisms, and advantages compared to traditional catalysts, incorporating principles of catalyst design.
    4.Please use professional scientific and chemical terminology to ensure that the generated analysis fully reflects a profound understanding of the chemical reaction.
    Now, give me your response:
"""

# EXTRACT_KONWLEDGE = """
# "{}. Based on the provided chemical reaction, extract relevant knowledge and concepts pertaining to functional groups, chemical reactions, and reaction mechanisms. 
# Summarize key information about the identified functional groups in both reactants and products, describe any changes in these functional groups during the reaction, and analyze their potential roles in the chemical reaction mechanism. 
# Your task is to facilitate the model in deriving ideal reaction conditions from the generated knowledge and concepts. Use concise and professional scientific terminology in your responses."
# """

EXTRACT_KNOWLEDGE = """
{}. Based on the provided chemical reaction, extract relevant knowledge and concepts.
You must generate your response in accordance with the provided format below.

Reactants: 
Reactant 1: []. functional group 1: [], functional group 2: []..., functional group n: [].
Reactant 2: []. functional group 1: [], functional group 2: []..., functional group n: [].
...

Products:
Product 1: []. functional group 1: [], functional group 2: [],..., functional group n: [].
Product 2: []. functional group 1: [], functional group 2: [],..., functional group n: [].
...

Reaction types:
Reaction type 1: [].
Reaction type 2: [].
...

Now, give me your response:
"""

PARSING = """
{}. Based on the provided chemical reaction, extract relevant knowledge and concepts.
You must generate your response in accordance with the provided format below. 
Notice that if there is only one reactant, list only 'Reactant 1'. 
The functional groups need to be specified with their respective molecular formulas.
Distinct functional groups, reactants, and reaction types need to be indicated using numbers.

Reactants: 
Reactant 1: []. functional group 1: [], functional group 2: []..., functional group n: [].
Reactant 2: []. functional group 1: [], functional group 2: []..., functional group n: [].

Products:
Product 1: []. functional group 1: [], functional group 2: [],..., functional group n: [].

Reaction types:
Reaction type 1: [].
Reaction type 2: [].
...

Now, give me your response:
"""

SYSTEM_PROMPT = "You are a chemical expert. Our goal is preparing some useful knowledge for chemical reaction condition recommendation task."