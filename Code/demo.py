import gradio as gr
from models import *

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     messages= history_openai_format,         
    #     temperature=1.0,
    #     stream=True
    # )
    response = solve_with_theorems(model='gpt-3.5-turbo', messages=history_openai_format)
    
    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message 

gr.ChatInterface(predict).queue().launch()