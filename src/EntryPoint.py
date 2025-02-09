from core import *

class SimpleWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Tkinter Window")
        self.root.geometry("300x150")

        # Create a text input field
        self.text_input = tk.Entry(root, width=30)
        self.text_input.pack(pady=10)

        # Create a button
        self.button = tk.Button(root, text="Submit", command=self.on_button_click)
        self.button.pack(pady=10)

        # Create a label to display the text
        self.output_label = tk.Label(root, text="", fg="blue")
        self.output_label.pack(pady=10)

    def on_button_click(self):
        # Get the text from the input field
        input_text = self.text_input.get()

        # Display the text in the label
        if input_text.strip():  # Check if the input is not empty
            self.output_label.config(text=f"You entered: {input_text}")
        else:
            messagebox.showwarning("Empty Input", "Please enter some text!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleWindow(root)
    root.mainloop()