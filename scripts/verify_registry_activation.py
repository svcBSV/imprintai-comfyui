"""Require the configured Comfy Registry node version to be active."""

import json
import pathlib
import sys
import tomllib
import urllib.request


ROOT = pathlib.Path(__file__).parents[1]
API_ROOT = "https://api.comfy.org/nodes"


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    release = json.loads((ROOT / "release.json").read_text())
    node_id = pyproject["project"]["name"]
    expected_version = pyproject["project"]["version"]

    node = fetch_json(f"{API_ROOT}/{node_id}")
    versions = fetch_json(f"{API_ROOT}/{node_id}/versions")
    version = next(
        (item for item in versions if item["version"] == expected_version),
        None,
    )

    if node.get("status") != "NodeStatusActive":
        raise ValueError(f"Registry node is not active: {node.get('status')}")
    if node.get("repository") != release["repository"]:
        raise ValueError("Registry listing points to an unexpected repository")
    if version is not None and version.get("status") == "NodeVersionStatusActive":
        if not version.get("downloadUrl"):
            raise ValueError(
                f"Active Registry version {expected_version} has no download URL"
            )
        print(f"Verified active Registry release {node_id} {expected_version}")
    else:
        print(f"Verified active Registry Git listing {node_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())