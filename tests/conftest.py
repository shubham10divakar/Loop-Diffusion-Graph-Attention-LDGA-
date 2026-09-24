import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # loop_vit.py, data.py
sys.path.insert(0, HERE)                    # loop_vit_reference.py (the original model, for T1/T2)
