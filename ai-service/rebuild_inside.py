"""Rebuild artifacts inside the running container."""
import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/app")
os.chdir("/app")

import subprocess
print("Upgrading scikit-learn...")
r = subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn>=1.6.0,<1.8.0", "-q"], capture_output=True, text=True)
if r.returncode != 0:
    print("PIP ERROR:", r.stderr[-500:])
else:
    print("Done.")

import sklearn
print(f"sklearn version: {sklearn.__version__}")

print("Running pipeline...")
from app.scripts.run_pipeline import main
main()

print("New artifacts:")
for f in sorted(os.listdir("/app/models")):
    if f.endswith(".pkl"):
        import joblib
        d = joblib.load(f"/app/models/{f}")
        print(f"  {f}: iv_features={len(d.get('iv_features', []))}, metrics={'YES' if 'metrics' in d else 'NO'}")

print("SCRIPT COMPLETE")
