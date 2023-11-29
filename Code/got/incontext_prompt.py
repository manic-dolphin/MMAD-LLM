SCORE_EXAMPLES_PROMPTS = """
Here are a few examples of ratings. Please strictly follow the provided format in returning the results.
Example 1: 
Input: All even numbers can be divided by 2.
Output: Score: [9]

Example 2:
Input: The square root of any integer is also an integer.
Output: Score: [1]

Example 3:
Input: Division can be performed when the divisor is zero.
Output: Score: [0]

Example 4:
Input: Every continuous function is differentiable.
Output: Score: [2]

Now, considering the initial solution along with the original problem, 
score the current solution on a scale of 0 to 10.
Only need to return the output, which is the score.
Input: the initial problem is {}. Here is the current solution: {}.
"""