from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import json

models_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(models_id)


# tokenizers: converts input string to list of integer tokens that would be input to a lm

input_prompt  = ["Hello, how did you convince the world that the god exists ?", "Hello, my cat is cute"]

tokenized_input = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to("mps")

print(tokenized_input)

print(tokenizer.batch_decode(tokenized_input["input_ids"]))

# input_ids & attention_mask 
# attention mask: soemthing like [0,0,0,1,1,1,1,1] where 0 means the token is padding and 1 means the token is not padding
tokenized_input.keys()