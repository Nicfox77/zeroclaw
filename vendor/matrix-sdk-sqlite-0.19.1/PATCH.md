# Patched bundled SQLite compatibility

Source: crates.io `matrix-sdk-sqlite` 0.19.1, Apache-2.0, from
<https://github.com/matrix-org/matrix-rust-sdk>.

The only functional change is the rusqlite requirement from 0.40.2 to 0.41
in the package manifests. ZeroClaw uses the same rusqlite generation throughout
the workspace so that exactly one SQLite native library is linked.

rusqlite 0.41 uses libsqlite3-sys 0.39 and patched SQLite, addressing
[RUSTSEC-2026-0459](https://rustsec.org/advisories/RUSTSEC-2026-0459.html).
The published Matrix SDK still requires rusqlite 0.40.2. Remove this patch when
a released Matrix SDK accepts rusqlite 0.41 or newer. No Matrix storage logic or
schema has been changed.

Registry cache markers and the upstream standalone Cargo.lock are excluded.
