"""Hydrate Cargo's cache from Terrane locks before offline native projection.

Only build-time downloads are prepared here; Terrane owns projection and compilation.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def prefetch(lock_path: Path) -> None:
    lock = tomllib.loads(lock_path.read_text())
    packages = [package for package in lock["package"] if "source" in package]
    with tempfile.TemporaryDirectory(prefix="cusco-cargo-fetch-") as temporary:
        root = Path(temporary)
        manifest = ['[package]', 'name = "cusco-cache-fetch"', 'version = "0.0.0"',
                    'edition = "2024"', '[workspace]', '[lib]', 'path = "lib.rs"',
                    '[dependencies]']
        for index, package in enumerate(packages):
            if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
                raise ValueError(f"Unsupported locked source: {package['source']}")
            requirement = "=" + package["version"].split("+", 1)[0]
            manifest.append(
                f'crate{index} = {{ package = {json.dumps(package["name"])}, '
                f'version = {json.dumps(requirement)}, default-features = false }}'
            )
        (root / "Cargo.toml").write_text("\n".join(manifest) + "\n")
        (root / "lib.rs").write_text("")
        subprocess.run(["cargo", "fetch", "--manifest-path", str(root / "Cargo.toml")], check=True)
        fetched = tomllib.loads((root / "Cargo.lock").read_text())
        checksums = {(p["name"], p["version"], p["source"]): p["checksum"]
                     for p in fetched["package"] if "source" in p}
        for package in packages:
            identity = (package["name"], package["version"], package["source"])
            if checksums[identity] != package["checksum"]:
                raise ValueError(f"Locked checksum mismatch: {identity}")
    print(f"Fetched and checked {len(packages)} locked crates from {lock_path}", flush=True)


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        prefetch(Path(argument))
