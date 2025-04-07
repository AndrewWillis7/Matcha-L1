import customtkinter as ctk
import sys
import os
import json

# Add parent dir to import message.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from message import Message

import api

def create_window(sizeX=500, sizeY=700):
    app = ctk.CTk()
    app.geometry(f"{sizeX}x{sizeY}")
    app.title("Matcha-L1 - Chatbot")

    # Theme settings
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("./src/colors.json")

    # Chat display
    chat_frame = ctk.CTkScrollableFrame(master=app, width=450, height=550)
    chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

    # Input frame
    input_frame = ctk.CTkFrame(master=app)
    input_frame.pack(fill="x", padx=10, pady=10)
    input_frame.grid_columnconfigure(0, weight=1)

    # Message entry
    entry = ctk.CTkEntry(master=input_frame, placeholder_text="Type a message...")
    entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    # Send button
    send_button = ctk.CTkButton(master=input_frame, text="Send", corner_radius=6,
                                command=lambda: send_message(entry, chat_frame))
    send_button.grid(row=0, column=1, padx=5, pady=10)

    # Sentiment viewer button
    sentiment_button = ctk.CTkButton(master=input_frame, text="Sentiment", corner_radius=6,
                                     command=show_sentiment_viewer)
    sentiment_button.grid(row=1, column=0, columnspan=2, pady=(0, 10))

    return app


def send_message(entry, chat_frame):
    user_text = entry.get().strip()
    if user_text:
        # Display user message
        user_label = ctk.CTkLabel(master=chat_frame, text=user_text, wraplength=380,
                                  corner_radius=6, fg_color="#33a8a3", text_color="#ffffff")
        user_label.pack(pady=2, padx=5, anchor="e")

        entry.delete(0, 'end')

        # Bot response
        bot_response = generate_bot_response(user_text)

        bot_label = ctk.CTkLabel(master=chat_frame, text=bot_response, wraplength=380,
                                 corner_radius=6, fg_color="#05403F", text_color="#f6f6f6")
        bot_label.pack(pady=2, padx=5, anchor="w")

        # Save sentiment
        msg_obj = Message(user_text)
        msg_obj.save_to_json()


def generate_bot_response(user_input):
    model_path = "./lib/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = api.load_model(model_path)

    if model and tokenizer:
        return api.generate_text(model, tokenizer, user_input, lambda x: None, max_length=100)

    return "I'm not sure how to respond to that!"


def show_sentiment_viewer():
    sentiment_window = ctk.CTkToplevel()
    sentiment_window.title("Sentiment Viewer")
    sentiment_window.geometry("400x600")

    scroll_frame = ctk.CTkScrollableFrame(master=sentiment_window, width=380, height=580)
    scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)

    if os.path.exists("messages.json"):
        with open("messages.json", "r") as file:
            data = json.load(file)

        for msg in data:
            sentiment = msg["sentiment"]["label"]
            text = msg["text"]
            color = "#1DB954" if sentiment == "positive" else "#E63946"

            bubble = ctk.CTkLabel(master=scroll_frame,
                                  text=text,
                                  wraplength=340,
                                  fg_color=color,
                                  corner_radius=10,
                                  text_color="#ffffff",
                                  padx=10,
                                  pady=6)
            bubble.pack(pady=4, anchor="e" if sentiment == "positive" else "w")
    else:
        no_data_label = ctk.CTkLabel(master=scroll_frame, text="No sentiment data found.")
        no_data_label.pack(pady=10)


if __name__ == "__main__":
    app = create_window()
    app.mainloop()
