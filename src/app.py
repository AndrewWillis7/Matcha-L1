import customtkinter as ctk

class ChatApp:
    def __init__(self, model, tokenizer, device, on_generate_function=None, sizeX=500, sizeY=700):
        """
        Initialize the ChatApp with the model, tokenizer, device, and an optional on_generate function.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.on_generate_function = on_generate_function

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
        """
        bot_label = ctk.CTkLabel(master=self.chat_frame, text=response, wraplength=380,
                                 corner_radius=6, fg_color="#05403F", text_color="#f6f6f6")
        bot_label.pack(pady=2, padx=5, anchor="w")

    def send_message(self):
        """
        Handle user messages and trigger text generation.
        """
        user_text = self.entry.get().strip()
        if user_text:
            # Display user message
            user_label = ctk.CTkLabel(master=self.chat_frame, text=f"{user_text}", wraplength=380,
                                     corner_radius=6, fg_color="#33a8a3", text_color="#ffffff")
            user_label.pack(pady=2, padx=5, anchor="e")

            self.entry.delete(0, 'end')  # Clear the input field

            # Display "Generating..." message
            generation_label = ctk.CTkLabel(master=self.chat_frame, text="Generating. . .", wraplength=380,
                                           corner_radius=6, fg_color="#05403F", text_color="#f6f6f6")
            generation_label.pack(pady=2, padx=5, anchor="w")

            # Call on_generate with the user's input and reply as the callback
            if self.on_generate_function:
                self.on_generate_function(user_text, callback=self.reply)

    def run(self):
        self.app.mainloop()