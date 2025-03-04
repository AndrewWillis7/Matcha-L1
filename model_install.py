from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Downloading Model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./lib")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./lib")

print("Downloaded Model and tokenizer!!")