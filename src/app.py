import customtkinter as ctk

class ChatApp:
    def __init__(self, model, tokenizer, device, on_generate_function=None, sizeX=500, sizeY=700, streaming=False):
        """
        Initialize the ChatApp with the model, tokenizer, device, and an optional on_generate function.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.streaming = streaming
        self.on_generate_function = on_generate_function
        self._streaming_label_created = False
        self.bot_label = None

        self.app = ctk.CTk()
        self.app.geometry(f"{sizeX}x{sizeY}")
        self.app.title("Matcha-L1 - Chatbot")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("./src/colors.json")

        self.chat_frame = ctk.CTkScrollableFrame(master=self.app, width=450, height=550)
        self.chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

        input_frame = ctk.CTkFrame(master=self.app)
        input_frame.pack(fill="x", padx=10, pady=10)

        self.entry = ctk.CTkEntry(master=input_frame, placeholder_text="Type a message...", width=350)
        self.entry.pack(side="left", padx=10, pady=10)

        send_button = ctk.CTkButton(master=input_frame, text="Send", corner_radius=6,
                                    command=self.send_message)
        send_button.pack(side="right", padx=10, pady=10)

    def reply(self, response):
        """
        Display the bot's response in the chat frame.
        - When streaming: creates ONE label on first chunk, then updates it
        - When not streaming: creates a new label with the complete response
        """
        if not self.streaming:
            # Non-streaming mode - create a complete new label
            bot_label = ctk.CTkLabel(
                master=self.chat_frame, 
                text=response, 
                wraplength=380,
                corner_radius=6, 
                fg_color="#05403F", 
                text_color="#f6f6f6"
            )
            bot_label.pack(pady=2, padx=5, anchor="w")
        else:
            # Streaming mode
            if not hasattr(self, '_streaming_label_created') or not self._streaming_label_created:
                # First streaming chunk - create the label
                self.bot_label = ctk.CTkLabel(
                    master=self.chat_frame, 
                    text=response,  # Initial text
                    wraplength=380,
                    corner_radius=6, 
                    fg_color="#71706f", 
                    text_color="#f6f6f6"
                )
                self.bot_label.pack(pady=2, padx=5, anchor="w")
                self._streaming_label_created = True  # Mark as created
            else:
                # Subsequent chunks - update existing label
                self.bot_label.configure(text=response)
                self.bot_label.update()  # Force GUI refresh
    
    def update_message_color(self):
        #05403F
        if self.bot_label:
            self.bot_label.configure(fg_color="#05403F")

    def send_message(self):
        """
        Handle user messages and trigger text generation.
        """
        if self._streaming_label_created:
            self._streaming_label_created = False

        user_text = self.entry.get().strip()
        if user_text:
            # Display user message
            user_label = ctk.CTkLabel(master=self.chat_frame, text=f"{user_text}", wraplength=380,
                                     corner_radius=6, fg_color="#33a8a3", text_color="#ffffff")
            user_label.pack(pady=2, padx=5, anchor="e")

            self.entry.delete(0, 'end')  # Clear the input field

            # Display "Generating..." message
            generation_label = ctk.CTkLabel(master=self.chat_frame, text="Generating. . .", wraplength=380,
                                           corner_radius=6, fg_color="#ff5733", text_color="#f6f6f6")
            generation_label.pack(pady=2, padx=5, anchor="w")

            # Call on_generate with the user's input and reply as the callback
            if self.on_generate_function:
                self.on_generate_function(user_text, callback=self.reply)

    def run(self):
        self.app.mainloop()