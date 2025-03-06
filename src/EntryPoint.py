from core import *

def main():
    model, tokens, device = api.load_model("./lib", use_onnx = True)
    ctk.set_default_color_theme("./src/colors.json")

    if model and tokens:
        app = api.create_ui(model, tokens, device_or_session=device)
        #print(f"Loaded UI and modelID: {model}")
        app.mainloop()
    else:
        print("CRITICAL ERROR, RESET!!")

main()

