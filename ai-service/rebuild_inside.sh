#!/usr/bin/env python3
"""Rebuild artifacts inside the running container using current code + updated requirements."""
import sys
import os
import warnings

# Suppress sklearn version mismatch warnings (will fix by retraining)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Ensure app is on path
sys.path.insert(0, "/app")
os.chdir("/app")

print("Upgrading scikit-learn...")
import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "install", "scikit-learn>=1.6.0,<1.8.0", "-q"],
    check=True,
)

# Patch requirements.txt in-place so rebuild is reproducible
with open("/app/requirements.txt") as f:
    content = f.read()
content = content.replace("scikit-learn>=1.4.0,<1.7.0", "scikit-learn>=1.6.0,<1.8.0")
with open("/app/requirements.txt", "w") as f:
    f.write(content)

print("Running pipeline to rebuild artifacts...")
from app.scripts.run_pipeline import main
main()

print("Done. New artifacts in /app/models/")
import os
for f in sorted(os.listdir("/app/models")):
    print(f"  {f}")
