from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Instantiate Model
model_name = "./lib/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set Model Device (If user has GPU then use GPU acceleration)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate API constants
app = FastAPI()

class TextGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


# Gonna try a different approach, dont like the API relying on a webhook . . . 