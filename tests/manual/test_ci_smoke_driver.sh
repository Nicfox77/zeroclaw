#!/usr/bin/env bash
# Exercise the actual smoke entrypoint with each builder capability boundary.
set -euo pipefail
smoke_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
smoke_fixture="$(mktemp -d)"
trap 'rm -rf "$smoke_fixture"' EXIT
mkdir -p "$smoke_fixture/bin" "$smoke_fixture/cache"
printf '{}\n' > "$smoke_fixture/cache/index.json"
cat > "$smoke_fixture/bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == 'buildx version' ]]; then exit 0; fi
if [[ "$1 $2" == 'buildx inspect' ]]; then
  printf 'Name: fixture\nDriver: %s\n' "$SMOKE_TEST_DRIVER"
  exit 0
fi
if [[ "$1 $2" == 'buildx build' ]]; then
  printf '%s\n' "$@" > "$SMOKE_TEST_ARGS"
  exit 0
fi
if [[ "$1" == run ]]; then exit 0; fi
exit 90
SH
chmod +x "$smoke_fixture/bin/docker"
cd "$smoke_repo"
for smoke_driver in docker docker-container; do
  PATH="$smoke_fixture/bin:$PATH" SMOKE_CACHE_DIR="$smoke_fixture/cache" \
    SMOKE_TEST_DRIVER="$smoke_driver" SMOKE_TEST_ARGS="$smoke_fixture/args" \
    ./dev/ci.sh docker-smoke
  if [[ "$smoke_driver" == docker ]]; then
    ! grep -q -- '--cache-' "$smoke_fixture/args"
  else
    grep -qx -- '--cache-to' "$smoke_fixture/args"
    grep -qx -- '--cache-from' "$smoke_fixture/args"
  fi
  grep -qx -- '--load' "$smoke_fixture/args"
done
if PATH="$smoke_fixture/bin:$PATH" SMOKE_TEST_DRIVER='' \
  ./dev/ci.sh docker-smoke > "$smoke_fixture/error" 2>&1; then
  echo 'An unidentified builder must fail explicitly.' >&2
  exit 1
fi
echo 'Smoke driver tests passed.'
