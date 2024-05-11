import time
# import openai
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# login(token="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")

# Define a function that adds a delay to a Completion API call
def delayed_completion(model, tokenizer, messages, delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    output, error = None, None
    for _ in range(max_trials):
        try:
            
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            
            device = "cuda"
            
            model_inputs = encodeds.to(device)
            model.to(device)
            
            generated_ids = model.generate(model_inputs, **kwargs)
            output = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])[0]

            break
        except Exception as e:
            error = e
            pass
    return output, error

def prepare_data(prompt, datum):
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
    return prompt + input_txt + postfix

def post_process(completion, choices):
    # content = completion['choices'][0]['message']['content'].strip()
    content = completion.split('.')[0]
    result = []
    for choice in choices:
        choice_txt = "Choice" + choice
        if choice_txt in content:
            result.append(choice)
    return content, result

import os

def get_model(model_name):
    if model_name not in ['e5-mistral-7b-instruct', 'gritlm-7b', 'gte-large', 'mistral_7b']:
        raise ValueError(f"{model_name} is not existed")
    
    local_model_dir = '/home/jupyter-t.hongquan/shared/models/' + model_name

    
    if model_name == "mistral_7b":
        return AutoModelForCausalLM.from_pretrained(local_model_dir), AutoTokenizer.from_pretrained(local_model_dir)

    return AutoModel.from_pretrained(local_model_dir), AutoTokenizer.from_pretrained(local_model_dir)


        