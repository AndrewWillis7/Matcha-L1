import sys
import os
import tkinter as tk

from tkinter import messagebox

# Paths
c_tkinter_path = os.path.abspath("./lib/c_tkinter")
packaging_path = os.path.abspath("./lib/packaging/src")
darkdetect_path = os.path.abspath("./lib/darkdetect")
pillow_path = os.path.abspath("./lib/pillow/src")

# Appending
sys.path.append(c_tkinter_path)
sys.path.append(packaging_path)
sys.path.append(darkdetect_path)
sys.path.append(pillow_path)

# local Distros
import darkdetect
import packaging
import customtkinter


