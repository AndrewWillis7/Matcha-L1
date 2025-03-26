# Matcha-L1

# Goal
The goal of this project is to gather and analyze chatbot responses from DeepSeek for research and academic purposes. This includes exploring its conversational patterns, response quality, sentiment, engagement, and common word usage. By studying these aspects, we aim to gain insights into how DeepSeek interacts with users sentimentally, identifying the overall balance in positive and negative responses.

# Methods
The study will follow a structured NLP research pipeline

Data Collection: We will collect chatbot responses using different types of prompts, such as factual questions, casual conversations, and opinion-based inputs.

Sentiment Analysis: We will check if chatbot responses are positive or negative using sentiment analysis.

Word Analysis: We will find and track common words and phrases to understand language patterns and repetition in chatbot responses.

# References
We will use research papers to understand how chatbots like DeepSeek interact sentimentally, including their tendencies to respond with positive or negative tones.

# Startup

Run the build script to install all the local dependencies and local LLM (MUST HAVE PYTHON 3.11 (For DirectML))

After install, run EntryPoint.py and the program will start

Here is the local install code for an example of localization of the model:

```PYTHON
#from model_install
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Downloading Model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./lib")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./lib")

model.save_pretrained("./lib")
tokenizer.save_pretrained("./lib")

print("Downloaded Model and tokenizer!!")
```
