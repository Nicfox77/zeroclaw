#!/usr/bin/env bash
# Opt-in Intel XPU provisioning; no host packages or kernel drivers are changed.
set -euo pipefail
umask 077
hindsight_root="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/zeroclaw/hindsight}"
[[ "$hindsight_root" == /* && -d "$hindsight_root/decision-model/serving" ]] || {
  echo 'Run setup-hindsight.sh with an absolute installation directory first.' >&2
  exit 2
}
command -v uv >/dev/null
command -v dpkg-deb >/dev/null
uv venv --python 3.12 --allow-existing "$hindsight_root/decision-xpu-runtime"
uv pip install --python "$hindsight_root/decision-xpu-runtime/bin/python" \
  'torch==2.15.0+xpu' --index-url https://download.pytorch.org/whl/xpu
uv pip install --python "$hindsight_root/decision-xpu-runtime/bin/python" \
  'transformers==5.22.2' "$hindsight_root/decision-model/serving[server]"
"$hindsight_root/decision-xpu-runtime/bin/python" - "$hindsight_root" <<'PYTHON'
from pathlib import Path
from urllib.request import urlopen
import hashlib, os, subprocess, sys
root = Path(sys.argv[1])
private = root / 'intel-compute' / '26.40.40196.8'
private.mkdir(parents=True, exist_ok=True)
# Tested Intel user-space driver, compiler, GMM, and matching Level Zero loader.
packages = [
 ('intel/compute-runtime', '26.40.40196.8', 'libze-intel-gpu1_26.40.40196.8-0_amd64.deb', 'd9c86b2bb195baea6936c2708846cd1fceeadc39bc5e167f728858a5ca6fb86e'),
 ('intel/compute-runtime', '26.40.40196.8', 'libigdgmm12_22.10.2_amd64.deb', '053fceadb84a1d2f9a30a0d1bc9b16424b3f1a90cc49900fcd80aadcf25c8a4d'),
 ('intel/intel-graphics-compiler', 'v2.42.6', 'intel-igc-core-2_2.42.6+23070_amd64.deb', 'e20539632f2d99875d8f8b225d5da58b31f98baa54bb47c6328aa98ec473ab9d'),
 ('oneapi-src/level-zero', 'v1.34.0', 'libze1_1.34.0+u24.04_amd64.deb', '45210e4549cd965ad7b9f160eefe53cbcdba01af341bea7ac7ba0aedeeb613ba'),
]
for repo, tag, name, digest in packages:
    package = private / name
    if not package.exists():
        with urlopen(f'https://github.com/{repo}/releases/download/{tag}/{name}', timeout=120) as response:
            data = response.read(200_000_001)
        if len(data) > 200_000_000 or hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit('Intel runtime package checksum/size mismatch')
        temporary = package.with_suffix('.download')
        temporary.write_bytes(data)
        temporary.replace(package)
    if hashlib.sha256(package.read_bytes()).hexdigest() != digest:
        raise SystemExit('Cached Intel runtime package checksum mismatch')
    subprocess.run(['dpkg-deb', '-x', str(package), str(private / 'root')], check=True)
# Use the official serving implementation with the verified attention/precision
# settings. Do not patch or vendor upstream Hindsight Python code.
launcher = root / 'serve-decision-xpu.py'
launcher.write_text("""from pathlib import Path
import torch
from hindsight_decision.server import main
root = Path(__file__).resolve().parent
torch.set_num_threads(4)
if not torch.xpu.is_available():
    raise SystemExit('Configured Intel XPU is unavailable; no implicit CPU fallback')
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    main([str(root / 'decision-model'), '--host', '127.0.0.1', '--port', '8000',
          '--device', 'xpu', '--dtype', 'float32', '--max-request-tokens', '8192',
          '--api-key', (root / 'decision.key').read_text().strip()])
""")
os.chmod(launcher, 0o700)
(root / 'decision-xpu.env').write_text(
    f'LD_LIBRARY_PATH={private}/root/usr/lib/x86_64-linux-gnu\n'
    'ONEAPI_DEVICE_SELECTOR=level_zero:gpu\nHF_HUB_OFFLINE=1\n')
os.chmod(root / 'decision-xpu.env', 0o600)
print('Provisioned pinned Intel XPU runtime; use the documented XPU service settings.')
PYTHON
