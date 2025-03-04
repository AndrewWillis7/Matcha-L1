import sys
import os

import customtkinter as ctk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import wraps

import api

# VARIABLES
version = 0.1


