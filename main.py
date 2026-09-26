from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import json

models_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(models_id)

# LM: language model
# language model is a neural network 
model = AutoModelForCausalLM.from_pretrained(models_id, dtype="auto")

# pipeline: pass a string, outcome's generation
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

generated_text = generation_pipeline(
    "Once upon a time, in a land far away, there lived a wise old owl who",
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)
print(json.dumps(generated_text, indent=4, ensure_ascii=False))

# batch generation
batch_input = [
    "Whos your daddy?",
    "What is the meaning of life?",
    ]
generated_batch = generation_pipeline(
    batch_input,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)
print(json.dumps(generated_batch, indent=4, ensure_ascii=False))