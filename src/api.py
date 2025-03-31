import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, StoppingCriteriaList
import torch
import torch_directml
import message

DEFAULT_MAX_LENGTH: int = 1000

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, update_callback, skip_prefix=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.update_callback = update_callback
        self.skip_prefix = skip_prefix
        self.buffer = ""
        self.skip_done = False
        self.final_text = ""  # Store the final complete text

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if not self.skip_done and self.skip_prefix:
            if self.skip_prefix in text:
                text = text.split(self.skip_prefix)[-1]
                self.skip_done = True

        if self.skip_done:
            self.buffer += text
            self.final_text = self.buffer  # Always keep the complete text
            self.update_callback(self.buffer)

    def on_stream_end(self):
        # Send final update with the complete text
        self.update_callback(self.final_text, stream_end=True)

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # If any token in the stop sequence is generated, stop generation
        return input_ids[0][-1].item() in self.stop_token_ids

class llm_interface:
    def __init__(self, model_path: str, use_gpu: bool = True, streaming: bool = False):
        """
        Initialize the llm_interface class by loading the model, tokenizer, and device.
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.streaming = streaming  # New streaming parameter
        self.completion_event = threading.Event()
        self.model, self.tokenizer, self.device = self._load_model()

        # Add callback reference
        self.update_callback = None

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

    def set_callback(self, callback):
        """
        Register an external callback for streaming updates.
        """
        self.update_callback = callback

    def generate_text(self, prompt: str, max_length: int = DEFAULT_MAX_LENGTH,
                      temperature: float = 0.2, top_k: int = 200, top_p: float = 1.0):
        """
        Generate text from a given prompt using the loaded model, with optional streaming.
        """
        if not self.model or not self.tokenizer or not self.device:
            return "Model, tokenizer, or device not loaded correctly."
        
        # Only use the streamer if a callback is registered
        streamer = None
        if self.streaming and self.update_callback:
            streamer = CustomStreamer(self.tokenizer, self.update_callback, skip_prefix="AI:")
        
        system_prompt = (
                "You are a helpful and strict assistant. Your job is to respond to user queries "
                "in a friendly, concise, and informative manner. You do not engage in self-reflection or "
                "create internal dialogues. Respond only to the user's prompt, and do not continue the conversation "
                "unless prompted by the user. You always have enough details to provide an accurate answer, "
                "and you do not speculate or talk to yourself."
        )
        user_prompt = f"{system_prompt}\nUser: {prompt}\nAI:"
    
        stop_sequence = "User:"  # Stop sequence to trim output
        stop_token_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)

        stop_criteria = StopOnToken(stop_token_ids)

        try:
            inputs = self.tokenizer(user_prompt, return_tensors="pt").to(self.device)

            if self.streaming and streamer:
                print("Streaming mode enabled...")
                stream_output = ""
                for output in self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    attention_mask=inputs['attention_mask'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([stop_criteria])
                ):
                    # Get the final text from the streamer if available
                    if hasattr(streamer, 'final_text'):
                        generated_text = streamer.final_text
                    else:
                        generated_text = ""

            else:
                print("Regular mode enabled...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        attention_mask=inputs['attention_mask'],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([stop_criteria])
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

                curr_message = message.Message(generated_text)
                curr_message.save_to_json()

                return generated_text
            except Exception as e:
                print("CRITICAL ERROR IN GENERATING TEXT")
                if callback:
                    callback("An error occurred while generating text.")
                return ""

        print("Generating Response!!")
        threading.Thread(target=generate_in_thread, daemon=True).start()


