from core import *
from transformers import AutoConfig
from transformers import TextStreamer

api_version = 0.1
DEFAULT_MAX_LENGTH: int = 1000
screen_size_x: int = 400

#------------------------------------------ CLASSES -------------------------------------------------------------#
#------------------------------------------ CLASSES -------------------------------------------------------------#
#------------------------------------------ CLASSES -------------------------------------------------------------#

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, update_callback, skip_prefix=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.update_callback = update_callback
        self.skip_prefix = skip_prefix
        self.buffer = ""
        self.skip_done = False
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        This method is called whenever a new token is generated.
        """
        if not self.skip_done and self.skip_prefix:
            # Skip the System Prompt
            if self.skip_prefix in text:
                text = text.split(self.skip_prefix)[-1]
                self.skip_done = True
        
        if self.skip_done:
            self.buffer += text
            self.update_callback(text)

    def on_stream_end(self):
        """
        This method is called when streaming ends.
        """
        self.update_callback(self.buffer, stream_end=True)
        


#------------------------------------------ FUNCS -------------------------------------------------------------#
#------------------------------------------ FUNCS -------------------------------------------------------------#
#------------------------------------------ FUNCS -------------------------------------------------------------#
# Model Loader
def load_model(model_path: str):
    """
     Loads a model from the local disk and prepares it for inference.
    """
    try:
                
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        config.use_sliding_window = False 

        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Verify model
        model.eval()

        # Check for GPU usage
        if torch.cuda.is_available():
            model.to('cuda')
    
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Text Generation
def generate_text(model, tokenizer, prompt: str, max_length: int = DEFAULT_MAX_LENGTH, 
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
        # Streamer Stuff
        #streamer = CustomStreamer(tokenizer, update_callback, skip_prefix=system_prompt)

        inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
        print("Tokenized input:", inputs)

        #Ensure the pad token ID is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id  # Use EOS token as PAD token

        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], 
                                     attention_mask=attention_mask,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     no_repeat_ngram_size=2,
                                     #max_new_tokens=100,
                                     #num_return_sequences=3,
                                     early_stopping=True,
                                     #repetition_penalty=1.2,
                                     #length_penalty=1.5,
                                     max_length=max_length,
                                     num_beams = 10,
                                     temperature=temperature,
                                     top_k=top_k,
                                     top_p=top_p,
                                     do_sample=True,
                                     #streamer=streamer
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decoded Output
        #print(f"Decoded Output: {generated_text}")

        # Delete all AI prefixes and Pretensor Errors
        ai_response_prefix = "AI:"
        if ai_response_prefix in generated_text:
            ai_response_start = generated_text.find(ai_response_prefix)

            generated_text = generated_text[ai_response_start + len(ai_response_prefix):]

            if stop_sequence in generated_text:
                generated_text = generated_text.split(stop_sequence)[0]
        
        # Trimmed Output
        print(f"Decoded Output (Trimmed): {generated_text}")
        return generated_text.strip()
    except Exception as e:
        print(f"Error generating text: {e}")
        return "An Error Occured when generating text!!"
    


def on_generate_click(root, model, tokenizer, input_textbox, output_label, max_length: int = DEFAULT_MAX_LENGTH, temperature: float = 1.0):
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
                generated_text = generate_text(model, tokenizer, prompt, max_length=int(max_length), temperature=temperature, top_p=0.9)
                root.after(0, lambda: insert_text_safe(output_label, f"\n{generated_text}"))
                print("Threading Response was Successful")
            except Exception as e:
                root.after(0, lambda: insert_text_safe(output_label, f"ERROR: {e}"))

        print("Generating Response!!")

        threading.Thread(target=generate_in_thread, daemon=True).start()
        output_label.configure(state="disabled")

        print("Finished Generating Response!!")



def create_ui(model, tokenizer):
    root = ctk.CTk()
    root.geometry(f"{screen_size_x}x{2 * screen_size_x}")
    root.title(f"Matcha-L1 | {api_version}")
    input_textbox = ctk.CTkEntry(root, placeholder_text="Enter prompt:")
    input_textbox.pack(pady=20, padx=20)

    output_label = ctk.CTkTextbox(root, wrap=None, width=(screen_size_x * 0.85), font=("Helvetica", 12))
    output_label.pack(expand=True, fill="y", pady=20)

    generate_button = ctk.CTkButton(root, text="Generate Text",
                                    command=lambda: on_generate_click(root, model, tokenizer,
                                                                      input_textbox,
                                                                      output_label, temperature=0.7,
                                                                      max_length=2000))
    generate_button.pack(pady=20)
    return root

def render_latex(latex_str):
    """
    Render LaTeX content into an image.
    """

    # Create Matplot Figure
    fig = plt.figure(figsize=(6,1))
    fig.patch.set_facecolor('none')
    plt.axis('off')

    # Render Latex
    plt.text(0.1, 0.5, f"${latex_str}$", fontsize=14, color='black')

    # Convert Figure to an IMAGE
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    # Close the figure to free memory
    plt.close(fig)

    return image