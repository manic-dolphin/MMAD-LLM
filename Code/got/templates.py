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
The reaction conditions of this reaction are:
"""