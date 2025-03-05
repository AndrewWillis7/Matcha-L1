from core import *
from transformers import AutoConfig
from transformers import TextStreamer


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
def generate_text(model, tokenizer, prompt: str, update_callback, max_length: int = 50, 
                  temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0):
    """
    Generate text from a given prompt using the loaded model.
    """
    # Prompt Manipulation
    system_prompt = "You are a helpful and strict assistant. Your job is to respond to user queries in a friendly and informative manner. Respond only to the current prompt and do not continue the conversation."
    user_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"

    # Sequence Limiter
    stop_sequence = "User:"



    try:
        # Streamer Stuff
        streamer = CustomStreamer(tokenizer, update_callback, skip_prefix=system_prompt)

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
                                     #num_return_sequences=3,
                                     #early_stopping=True,
                                     repetition_penalty=1.2,
                                     #length_penalty=1.5,
                                     max_length=max_length,
                                     num_beams = 3,
                                     temperature=temperature,
                                     top_k=top_k,
                                     top_p=top_p,
                                     do_sample=True,
                                     streamer=streamer)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decoded Output
        print(f"Decoded Output: {generated_text}")

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
    


def on_generate_click(root, model, tokenizer, input_textbox, output_label, max_length: int = 50, temperature: float = 1.0):
    prompt = input_textbox.get()
    if prompt:
        output_label.configure(text="Generating. . .")

        # Set callback to update UI
        def update_callback(text: str, stream_end: bool = False):

            #if "\\" in text:
                 # Render LaTeX content
                #latex_image = render_latex(text)
                #latex_photo = ImageTk.PhotoImage(latex_image)

                # Display the rendered image
                #output_label.configure(image=latex_photo)
                #output_label.image = latex_photo
            #else:
            output_label.configure(text=output_label.cget("text") + text)
            root.update()

        generated_text = generate_text(model, tokenizer, prompt, update_callback, max_length=int(max_length), temperature=temperature, top_p=0.9)
        output_label.configure(text=generated_text)



def create_ui(model, tokenizer):
    root = ctk.CTk()
    root.title("Matcha-L1")
    input_textbox = ctk.CTkEntry(root, placeholder_text="Enter prompt:")
    input_textbox.pack(pady=20, padx=20)

    output_label = ctk.CTkLabel(root, text="", wraplength=400)
    output_label.pack(pady=20, padx=20)

    generate_button = ctk.CTkButton(root, text="Generate Text",
                                    command=lambda: on_generate_click(root, model, tokenizer,
                                                                      input_textbox,
                                                                      output_label, temperature=0.1,
                                                                      max_length=200))
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