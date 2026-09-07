import os
import sys

# Sandbox-safe path setup (avoid PYTHONPATH which clobbers injected runfiles).
_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_ROOT, "src"))
# M6 moved metrics to the top-level eval/ package PLAN.md specifies (see
# M6_ONBOARDING.md §4), so `import eval.metrics` must resolve from the repo root.
sys.path.insert(0, _ROOT)
