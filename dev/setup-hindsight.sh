#!/usr/bin/env bash
# Explicit, CPU-only provisioning. Runtime startup never installs packages or weights.
set -euo pipefail
umask 077
hindsight_root="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/zeroclaw/hindsight}"
if [[ "$hindsight_root" != /* ]]; then
  echo 'Hindsight installation directory must be absolute.' >&2
  exit 2
fi
command -v uv >/dev/null
command -v hf >/dev/null
mkdir -p "$hindsight_root"
chmod 700 "$hindsight_root"
if [[ ! -x "$hindsight_root/runtime/bin/python" ]]; then
  uv venv --python 3.12 "$hindsight_root/runtime"
fi
uv pip install --python "$hindsight_root/runtime/bin/python" \
  'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$hindsight_root/runtime/bin/python" \
  'hindsight-embed==1.2.0' 'hindsight-api==1.2.0' \
  'transformers==5.15.1' 'sentence-transformers==6.2.0' 'pg0-embedded==0.16.1'
# Download by immutable revision. Preserve HF's cache layout for offline startup.
HF_HOME="$hindsight_root/models" hf download hindsight-ai/hs-embed-1 \
  --revision 3164caa4db95a03faac8a327f914f7c8f42d2f98
HF_HOME="$hindsight_root/models" hf download hindsight-ai/hs-rerank-1 \
  --revision 927f5be31280298a51cf29af8a29cb111ed812c5
hf download hindsight-ai/hs-decide-1-flash \
  --revision e2c789026dfd1f7608cafb2ef6a9bb31397af802 \
  --local-dir "$hindsight_root/decision-model"
if [[ ! -x "$hindsight_root/decision-runtime/bin/python" ]]; then
  uv venv --python 3.12 "$hindsight_root/decision-runtime"
fi
uv pip install --python "$hindsight_root/decision-runtime/bin/python" \
  'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
# Official model-repository serving package, including /v1/systemone.
uv pip install --python "$hindsight_root/decision-runtime/bin/python" \
  'transformers==5.22.2' "$hindsight_root/decision-model/serving[server]"
"$hindsight_root/decision-runtime/bin/python" - "$hindsight_root" <<'PY'
from pathlib import Path
import os
import secrets
import sys
root = Path(sys.argv[1])
key = root / 'decision.key'
if not key.exists():
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(secrets.token_urlsafe(32))
os.chmod(key, 0o600)
# Thin launcher only; all inference and HTTP behavior stays in upstream's package.
launcher = root / 'serve-decision.py'
launcher.write_text('''from pathlib import Path
from hindsight_decision.server import main
root = Path(__file__).resolve().parent
main([str(root / 'decision-model'), '--host', '127.0.0.1', '--port', '8000',
      '--device', 'cpu', '--dtype', 'float32', '--max-request-tokens', '8192',
      '--api-key', (root / 'decision.key').read_text().strip()])
''')
os.chmod(launcher, 0o700)
PY
# The optional official Control Plane is installed once, never fetched by startup.
if [[ "${ZEROCLAW_HINDSIGHT_WITH_UI:-0}" == 1 ]]; then
  command -v npm >/dev/null
  npm install --prefix "$hindsight_root/control-plane" --save-exact --ignore-scripts \
    --no-audit --no-fund '@vectorize-io/hindsight-control-plane@1.2.0'
fi
printf 'Provisioned pinned CPU runtime and model cache in %s\n' "$hindsight_root"
printf 'Configure memory.hindsight, then run: zeroclaw memory setup\n'
printf 'Start the separate decision server using the service example in docs/book/src/agents/hindsight.md\n'
