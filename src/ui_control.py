from core import *

def create_window(sizeX=500, sizeY=700):
    app = customtkinter.CTk()
    app.geometry(f"{sizeX}x{sizeY}")
    app.title(f"Matcha-L1  - DEVELOPMENT COPY  -  Version: {version}")

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("./src/colors.json")

    button = customtkinter.CTkButton(master=app, text="Open Log", corner_radius=32)
    button.place(relx=0.5, rely=0.5, anchor="center")

    entry = customtkinter.CTkEntry(master=app, placeholder_text="Start Typing...", width=(sizeX*0.75))
    entry.place(relx=0.5, rely=0.9, anchor="center")

    return app