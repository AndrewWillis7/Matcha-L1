from core import *

model, tokens = api.load_model("./lib")

app = api.create_ui(model, tokens)
app.mainloop()
