from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import json

models_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(models_id)

# isntruction prompts / chat templates
# Instruction tuning: LMs are finetuned to follow user instructions, in a chat like format

prompt_template = [
    {
        "role": "system",
        "content": "You are a Smart A Assistant Who Speaks Like a Pirate"
    },
    {
        "role": "user",
        "content": "Where does the sun rises?"
    }
]


tokenized = tokenizer.apply_chat_template(prompt_template, padding=True, truncation=True, return_tensors="pt").to("mps")

print(tokenized)