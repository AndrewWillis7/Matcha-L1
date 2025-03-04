from core import *

# Model Loader
def load_model(model_path: str):
    """
     Loads a model from the local disk and prepares it for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Verify model
    model.eval()

    # Check for GPU usage
    if torch.cuda.is_available():
        model.to('cuda')
    
    return model, tokenizer

# Text Generation
def generate_text(model, tokenizer, prompt: str, max_length: int = 50):
    """
    Generate text from a given prompt using the loaded model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=max_length)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def on_generate_click(model, tokenizer, input_textbox, output_label):
    prompt = input_textbox.get()
    generated_text = generate_text(model, tokenizer, prompt)
    output_label.configure(text=generated_text)

def create_ui(model, tokenizer):
    root = ctk.CTk()
    root.title("Matcha-L1")
    input_textbox = ctk.CTkEntry(root, placeholder_text="Enter prompt:")
    input_textbox.pack(pady=20, padx=20)

    output_label = ctk.CTkLabel(root, text="", wraplength=1000)
    output_label.pack(pady=20, padx=20)

    generate_button = ctk.CTkButton(root, text="Generate Text",
                                    command=lambda: on_generate_click(model, tokenizer,
                                                                      input_textbox,
                                                                      output_label))
    generate_button.pack(pady=20)
    return root
