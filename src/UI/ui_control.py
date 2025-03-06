# from core import *

# def create_window(sizeX=500, sizeY=700):
#     app = customtkinter.CTk()
#     app.geometry(f"{sizeX}x{sizeY}")
#     app.title(f"Matcha-L1  - DEVELOPMENT COPY  -  Version: {version}")

#     customtkinter.set_appearance_mode("dark")
#     customtkinter.set_default_color_theme("./src/colors.json")

#     button = customtkinter.CTkButton(master=app, text="Open Log", corner_radius=32)
#     button.place(relx=0.5, rely=0.5, anchor="center")

#     entry = customtkinter.CTkEntry(master=app, placeholder_text="Start Typing...", width=(sizeX*0.75))
#     entry.place(relx=0.5, rely=0.9, anchor="center")

#     return app



import customtkinter as ctk
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api

def create_window(sizeX=500, sizeY=700):
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    app = ctk.CTk()
    app.geometry(f"{sizeX}x{sizeY}")
    app.title("Matcha-L1 - Chatbot")

    # custom color settings
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("./src/colors.json")

    # scrollable chat display
    chat_frame = ctk.CTkScrollableFrame(master=app, width=450, height=550)
    chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

    # input frame
    input_frame = ctk.CTkFrame(master=app)
    input_frame.pack(fill="x", padx=10, pady=10)

    # box for user input
    entry = ctk.CTkEntry(master=input_frame, placeholder_text="Type a message...", width=350)
    entry.pack(side="left", padx=10, pady=10)

    # send button
    send_button = ctk.CTkButton(master=input_frame, text="Send", corner_radius=6,
                                command=lambda: send_message(entry, chat_frame))
    send_button.pack(side="right", padx=10, pady=10)

    return app

def send_message(entry, chat_frame):
    #Handles user messages and displaying bot responses
    
    user_text = entry.get().strip()
    if user_text:
        # displays user message
        user_label = ctk.CTkLabel(master=chat_frame, text=f"{user_text}", wraplength=380,
                                  corner_radius=6, fg_color="#33a8a3", text_color="#ffffff")
        user_label.pack(pady=2, padx=5, anchor="e")

        entry.delete(0, 'end')  # clears input on sending section

        # this will generate the bot response
        bot_response = generate_bot_response(user_text)
        
        # location where the actual chatbot response should go
        bot_label = ctk.CTkLabel(master=chat_frame, text=f"{bot_response}", wraplength=380,
                                 corner_radius=6, fg_color="#05403F", text_color="#f6f6f6")
        bot_label.pack(pady=2, padx=5, anchor="w")

def generate_bot_response(user_input):
    # Calls the API to generate a chatbot respone
    model_path = "./lib/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = api.load_model(model_path)

    if model and tokenizer:
        return api.generate_text(model, tokenizer, user_input, lambda x: None, max_length=100)
    
    # This is the placeholder response that should be replaced with actual chatbot output
    return "I'm not sure how to respond to that!"

if __name__ == "__main__":
    app = create_window()
    app.mainloop()
