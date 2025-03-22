import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_directml

DEFAULT_MAX_LENGTH: int = 1000

class llm_interface:
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the TextGenerator class by loading the model, tokenizer, and device.
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.completion_event = threading.Event()
        self.model, self.tokenizer, self.device = self._load_model()

    def _choose_device(self):
        """
        Choose the best device available (CUDA for NVIDIA, DirectML for AMD, or CPU).
        """
        if torch.cuda.is_available():
            print("Using NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        elif torch_directml.is_available():
            print(f"Using DirectML device: {torch_directml.device()}")
            return torch.device(torch_directml.device())
        else:
            print("Using CPU")
            return torch.device("cpu")

    def _load_model(self):
        """
        Load the model, tokenizer, and move the model to the appropriate device.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            device = self._choose_device() if self.use_gpu else torch.device("cpu")
            model.to(device)
            print(f"Loaded model on {device}")
            return model, tokenizer, device
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None

    def generate_text(self, prompt: str, max_length: int = DEFAULT_MAX_LENGTH,
                      temperature: float = 1.0, top_k: int = 200, top_p: float = 1.0):
        """
        Generate text from a given prompt using the loaded model.
        """
        if not self.model or not self.tokenizer or not self.device:
            return "Model, tokenizer, or device not loaded correctly."

        system_prompt = (
            "You are a helpful and strict assistant. Your job is to respond to user queries "
            "in a friendly and informative manner. Respond only to the current prompt and do not "
            "continue the conversation. You always have enough details and can provide an accurate answer."
        )
        user_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"
        stop_sequence = "User:"  # Stop sequence to trim output

        try:
            inputs = self.tokenizer(user_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    early_stopping=True,
                    attention_mask=inputs['attention_mask'],
                    num_beams=10,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Post-process the generated text
            ai_response_prefix = "AI:"
            if ai_response_prefix in generated_text:
                ai_response_start = generated_text.find(ai_response_prefix)
                generated_text = generated_text[ai_response_start + len(ai_response_prefix):]

                if stop_sequence in generated_text:
                    generated_text = generated_text.split(stop_sequence)[0]

            self.completion_event.set()
            return generated_text.strip()

        except Exception as e:
            print(f"Error generating text: {e}")
            return "An error occurred when generating text!"

    def on_generate(self, prompt, callback=None, max_length: int = DEFAULT_MAX_LENGTH, temperature: float = 1.0):
        """
        Generate text in a background thread and handle the result.
        If a callback is provided, it will be called with the generated text.
        """
        if not prompt:
            return

        def generate_in_thread():
            try:
                generated_text = self.generate_text(
                    prompt, max_length=max_length, temperature=temperature, top_p=0.9
                )
                print("Threading Response was Successful\n")
                print(f"{generated_text}\n")
            
                # Call the callback with the generated text
                if callback:
                    callback(generated_text)
                return generated_text
            except Exception as e:
                print("CRITICAL ERROR IN GENERATING TEXT")
                if callback:
                    callback("An error occurred while generating text.")
                return ""

        print("Generating Response!!")
        threading.Thread(target=generate_in_thread, daemon=True).start()