from api import llm_interface
from app import ChatApp

class Runtime:
    def __init__(self, model_path: str, use_gpu: bool = True, streaming_mode: bool = True):
        """
        Initialize the MainApp class with the LLM interface and ChatApp UI.
        """
        self.ai = llm_interface(model_path, use_gpu, streaming=streaming_mode)
        self.chat_app = ChatApp(model=self.ai.model, tokenizer=self.ai.tokenizer, device=self.ai.device, on_generate_function=self.ai.on_generate, streaming=streaming_mode)
        
        # Register the callback after ChatApp initialization
        self.ai.set_streaming_callback(self.chat_app.reply)

    def run(self):
        """
        Run the application if the model and tokenizer are loaded successfully.
        """
        if self.ai.model and self.ai.tokenizer:
            print("Application started successfully.")
            self.chat_app.run()
        else:
            print("CRITICAL ERROR: Failed to load model or tokenizer. Exiting...")

def main():
    # Initialize and run the main application
    app = Runtime("./lib", use_gpu=False, streaming_mode=True)
    app.run()

if __name__ == "__main__":
    main()