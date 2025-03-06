import sys
import os
import threading
import time

import customtkinter as ctk
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import wraps

import api