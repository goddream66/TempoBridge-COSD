"""Load the built wheel and exported encoder outside the repository on CPU."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", default="runs/portability.json")
    parser.add_argument("--temp-root", help="Optional writable parent for the disposable relocation check")
    args = parser.parse_args()
    wheel, encoder = Path(args.wheel).resolve(), Path(args.encoder).resolve()
    temp_root = Path(args.temp_root).resolve() if args.temp_root else None
    if temp_root:
        temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tempobridge_portable_", dir=temp_root) as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                resolved = (root / name).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("Wheel member escapes relocation directory")
            archive.extractall(root)
        shutil.copy2(encoder, root / "encoder.pt")
        code = """
import json
from pathlib import Path
import torch
import tempobridge
from tempobridge.inference import load_encoder
from tempobridge.utils import load_checkpoint
assert Path(tempobridge.__file__).resolve().is_relative_to(Path.cwd())
torch.set_num_threads(2)
state = load_checkpoint('encoder.pt')
cfg = state['data_config']
length = round(max(cfg['scales_seconds']) * cfg['ppg_fs'])
encoder = load_encoder('encoder.pt')
with torch.inference_mode():
    features = encoder(torch.randn(2, 1, length))
assert torch.isfinite(features).all()
print(json.dumps({'relocated_wheel_import': True, 'relative_checkpoint_load': True,
                  'feature_shape': list(features.shape), 'device': 'cpu'}))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                check=True, capture_output=True, text=True)
        report = json.loads(result.stdout.strip())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
