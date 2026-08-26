import os
import sys

# Sandbox-safe path setup (avoid PYTHONPATH which clobbers injected runfiles).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
