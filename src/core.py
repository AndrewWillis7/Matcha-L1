import sys
import os

import tkinter as tk
from tkinter import messagebox

# Set up Paths

# ----------------------------------------------- LIB PATH -----------------------------------------------------
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
if lib_path not in sys.path:
    sys.path.append(lib_path)

print(lib_path)
# ----------------------------------------------- LIB PATH -----------------------------------------------------

c_tkinter_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "./lib/c_tkinter/", "customtkinter"))

if c_tkinter_path not in sys.path:
    sys.path.append(c_tkinter_path)

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "./lib/packaging/", "src"))

if package_path not in sys.path:
    sys.path.append(package_path)

darkdetect_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "./lib/darkdetect/", "darkdetect"))

if darkdetect_path not in sys.path:
    sys.path.append(darkdetect_path)


# Attempt Imports ------ > 

try:
    import packaging
    print("Import successful!")
except ModuleNotFoundError as e:
    print("Import failed:", e)

try:
    import c_tkinter.customtkinter
    print("Import successful!")
except ModuleNotFoundError as e:
    print("Import failed:", e)

try:
    import darkdetect
    print("Import successful!")
except ModuleNotFoundError as e:
    print("Import failed:", e)



#from lib.c_tkinter import customtkinter as ck

# Paths
#c_tkinter_path = os.path.abspath("./lib/c_tkinter")
#packaging_path = os.path.abspath("./lib/packaging/src")
#darkdetect_path = os.path.abspath("./lib/darkdetect")
#pillow_path = os.path.abspath("./lib/pillow/src")

# Appending
#sys.path.append(c_tkinter_path)
#sys.path.append(packaging_path)
#sys.path.append(darkdetect_path)
#sys.path.append(pillow_path)

# local Distros
#import darkdetect
#import packaging
#import customtkinter


