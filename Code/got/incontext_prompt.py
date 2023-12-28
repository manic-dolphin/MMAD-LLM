# SCORE_EXAMPLES_PROMPTS = """
# Here are a few examples of ratings. Please strictly follow the provided format in returning the results. Now, considering the initial generated reaction conditions along with the original problem, 
# score the current reaction conditions on a scale of 0 to 10. Notice that you absolutely cannot return anything other than score[] :

# All even numbers can be divided by 2.  => Score: [9]

# The square root of any integer is also an integer. => Score: [1]

# Division can be performed when the divisor is zero. => Score: [0]

# Every continuous function is differentiable. => Score: [2]

# The initial problem is {}. Here is the current reaction conditions: {}. => Score:
# """

SCORE_EXAMPLES_PROMPTS = """
"Combining the instruction {}, now I provide you with a generated response: {}. 
You're required to evaluate it based on its quality and reasonableness, using a scoring range from 0 to 10 combining with the instruction. Rating: "
"""