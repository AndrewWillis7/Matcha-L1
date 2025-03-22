import torch
import torch_directml
import threading
import customtkinter as ctk
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import uicontrol as GUI

api_version = 1.3
DEFAULT_MAX_LENGTH: int = 1000
screen_size_x: int = 400

MODEL_LOADED: bool = False
_model = None
_tokenizer = None

#------------------------------------------ FUNCS -------------------------------------------------------------#

def choose_device():
    """
    Choose the best device available (CUDA for NVIDIA, DirectML for AMD, or CPU).
    """
    if torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    elif torch_directml.is_available():
        torch_directml.gpu_memory
        print(f"Using DirectML device: {torch_directml.device()}")
        return torch.device(torch_directml.device())
    else:
        print("Using CPU")
        return torch.device("cpu")

def load_model(model_path: str, use_gpu: bool = True):
    """
    Load a model from the local disk and prepare it for inference.
    Selects device (GPU or CPU) based on availability.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        device = choose_device() if use_gpu else torch.device("cpu")
        model.to(device)
        print(f"Loaded model on {device}")
        
        MODEL_LOADED = True

        _model = model
        _tokenizer = tokenizer

        return model, tokenizer, device

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def get_model_properties():
    if (MODEL_LOADED):
        return _model, _tokenizer

def generate_text(model, tokenizer, device, prompt: str, max_length: int = DEFAULT_MAX_LENGTH,
                  temperature: float = 1.0, top_k: int = 200, top_p: float = 1.0):
    """
    Generate text from a given prompt using the loaded model on the given device.
    """
    system_prompt = "You are a helpful and strict assistant. Your job is to respond to user queries in a friendly and informative manner. Respond only to the current prompt and do not continue the conversation. You always have enough details and can provide an accurate answer."
    user_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"

    stop_sequence = "User:"  # Stop sequence to trim output

    try:
        inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                early_stopping=True,
                attention_mask=inputs['attention_mask'],
                num_beams=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the generated text
        ai_response_prefix = "AI:"
        if ai_response_prefix in generated_text:
            ai_response_start = generated_text.find(ai_response_prefix)
            generated_text = generated_text[ai_response_start + len(ai_response_prefix):]

            if stop_sequence in generated_text:
                generated_text = generated_text.split(stop_sequence)[0]

        return generated_text.strip()
    
    except Exception as e:
        print(f"Error generating text: {e}")
        return "An error occurred when generating text!"

def on_generate_click(root, model, tokenizer, device, input_textbox, output_label, max_length: int = DEFAULT_MAX_LENGTH, temperature: float = 1.0):
    """
    Handle the button click event, which generates text in a background thread.
    """
    prompt = input_textbox.get()
    if prompt:
        def insert_text_safe(widget, text):
            widget.configure(state="normal")
            widget.insert("end", text)  # Insert at the end instead of "1.0"
            widget.see("end")  # Auto-scroll to the latest line
            widget.configure(state="disabled")

        insert_text_safe(output_label, "\n\nGenerating Response. . . \n\n")

        def generate_in_thread():
            try:
                generated_text = generate_text(model, tokenizer, device, prompt, max_length=int(max_length), temperature=temperature, top_p=0.9)
                root.after(0, lambda: insert_text_safe(output_label, f"\n{generated_text}"))
                print("Threading Response was Successful")
            except Exception as e:
                root.after(0, lambda: insert_text_safe(output_label, f"ERROR: {e}"))

        print("Generating Response!!")
        threading.Thread(target=generate_in_thread, daemon=True).start()
        output_label.configure(state="disabled")

def create_ui():
    """
    Create the Tkinter UI and initialize necessary components.
    """

    """
    root = ctk.CTk()
    root.geometry(f"{screen_size_x}x{2 * screen_size_x}")
    root.title(f"Matcha-L1 | {api_version}")
    
    input_textbox = ctk.CTkEntry(root, placeholder_text="Enter prompt:")
    input_textbox.pack(pady=20, padx=20)

    output_label = ctk.CTkTextbox(root, wrap=None, width=(screen_size_x * 0.85), font=("Helvetica", 12))
    output_label.pack(expand=True, fill="y", pady=20)

    generate_button = ctk.CTkButton(root, text="Generate Text", command=lambda: on_generate_click(root, model, tokenizer, device, input_textbox, output_label, temperature=0.7, max_length=2000))
    generate_button.pack(pady=20)    
    """

    app = GUI.create_window()
    
    return app
