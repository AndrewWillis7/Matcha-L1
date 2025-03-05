import sys
import os
import threading
import time

import customtkinter as ctk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk

import api

# VARIABLES
version = 0.1


