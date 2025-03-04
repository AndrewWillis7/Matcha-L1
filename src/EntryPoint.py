from core import *

api.load_model("./lib")

app = api.create_ui()
app.mainloop()
