import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import threading
import customtkinter as ctk
import numpy as np
from pathlib import Path
from transformers.onnx import OnnxConfig, export

api_version = 0.1
DEFAULT_MAX_LENGTH: int = 1000
screen_size_x: int = 400

#------------------------------------------ FUNCS -------------------------------------------------------------#

def export_to_onnx(model, tokenizer, model_path: str, opset: int = 14):
    """
    Export a Hugging Face model to ONNX format.
    """
    onnx_path = Path("./lib/onnx_container/src")  # Convert to Path object
    print(f"Exporting model to ONNX format: {onnx_path}")

    # Define a custom ONNX configuration
    class CustomOnnxConfig(OnnxConfig):
        @property
        def inputs(self):
            return {"input_ids": {0: "batch", 1: "sequence"}}

        @property
        def outputs(self):
            return {"logits": {0: "batch", 1: "sequence"}}

    # Create the ONNX configuration
    onnx_config = CustomOnnxConfig(model.config)

    # Export the model
    export(
        preprocessor=tokenizer,  # Pass the tokenizer as the preprocessor
        model=model,
        config=onnx_config,
        opset=opset,
        output=onnx_path,  # Use Path object here
    )
    print("Model exported successfully!")
    return onnx_path

def load_onnx_model(onnx_path: str, use_gpu: bool = True):
    """
    Load an ONNX model using ONNX Runtime.
    """
    providers = ["CPUExecutionProvider"]  # Default to CPU

    if use_gpu:
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider"]  # Use NVIDIA GPU
        elif "DmlExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["DmlExecutionProvider"]  # Use AMD GPU (DirectML)
        elif "ROCMExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["ROCMExecutionProvider"]  # Use AMD GPU (ROCm)

    print(f"Using providers: {providers}")
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return session

def load_model(model_path: str, use_onnx: bool = True):
    """
    Load a model from the local disk and prepare it for inference.
    If use_onnx is True, the model will be exported to ONNX format (if not already) and loaded with ONNX Runtime.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.use_sliding_window = False

        if use_onnx:
            # Export the model to ONNX format (if not already exported)
            onnx_path = f"{model_path}.onnx"
            try:
                session = load_onnx_model(onnx_path)
            except:
                # Export the model to ONNX format
                model = AutoModelForCausalLM.from_pretrained(model_path)
                onnx_path = export_to_onnx(model, tokenizer, model_path)
                session = load_onnx_model(onnx_path)
            return session, tokenizer, "onnx"
        else:
            # Load the model in PyTorch format
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            return model, tokenizer, device

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def generate_text(model, tokenizer, device_or_session, prompt: str, max_length: int = DEFAULT_MAX_LENGTH,
                  temperature: float = 1.0, top_k: int = 200, top_p: float = 1.0):
    """
    Generate text from a given prompt using the loaded model.
    """
    # Prompt Manipulation
    system_prompt = "You are a helpful and strict assistant. Your job is to respond to user queries in a friendly and informative manner. Respond only to the current prompt and do not continue the conversation. You always have enough details and can provide an accurate answer."
    user_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"

    # Sequence Limiter
    stop_sequence = "User:"

    try:
        if isinstance(device_or_session, str) and device_or_session == "onnx":
            # ONNX Runtime inference
            inputs = tokenizer(user_prompt, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)

            # Run inference
            outputs = model.run(None, {"input_ids": input_ids})
            generated_ids = outputs[0]

            # Decode the generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # PyTorch inference
            inputs = tokenizer(user_prompt, return_tensors="pt").to(device_or_session)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
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
        return "An Error Occurred when generating text!!"

def on_generate_click(root, model, tokenizer, device_or_session, input_textbox, output_label, max_length: int = DEFAULT_MAX_LENGTH, temperature: float = 1.0):
    prompt = input_textbox.get()
    if prompt:
        # Helper function to safely update text in Tkinter
        def insert_text_safe(widget, text):
            widget.configure(state="normal")
            widget.insert("end", text)  # Insert at the end instead of "1.0"
            widget.see("end")  # Auto-scroll to the latest line
            widget.configure(state="disabled")

        insert_text_safe(output_label, "\nGenerating Response. . . \n\n")

        def generate_in_thread():
            try:
                generated_text = generate_text(model, tokenizer, device_or_session, prompt, max_length=int(max_length), temperature=temperature, top_p=0.9)
                root.after(0, lambda: insert_text_safe(output_label, f"\n{generated_text}"))
                print("Threading Response was Successful")
            except Exception as e:
                root.after(0, lambda: insert_text_safe(output_label, f"ERROR: {e}"))

        print("Generating Response!!")

        threading.Thread(target=generate_in_thread, daemon=True).start()
        output_label.configure(state="disabled")

        print("Finished Generating Response!!")

def create_ui(model, tokenizer, device_or_session):
    root = ctk.CTk()
    root.geometry(f"{screen_size_x}x{2 * screen_size_x}")
    root.title(f"Matcha-L1 | {api_version}")
    input_textbox = ctk.CTkEntry(root, placeholder_text="Enter prompt:")
    input_textbox.pack(pady=20, padx=20)

    output_label = ctk.CTkTextbox(root, wrap=None, width=(screen_size_x * 0.85), font=("Helvetica", 12))
    output_label.pack(expand=True, fill="y", pady=20)

    generate_button = ctk.CTkButton(root, text="Generate Text",
                                    command=lambda: on_generate_click(root, model, tokenizer, device_or_session,
                                                                      input_textbox,
                                                                      output_label, temperature=0.7,
                                                                      max_length=2000))
    generate_button.pack(pady=20)
    return root
