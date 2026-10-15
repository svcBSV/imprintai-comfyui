"""Validate release metadata without requiring ComfyUI."""

import json
import pathlib
import re
import sys
import tomllib


ROOT = pathlib.Path(__file__).parents[1]


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    release = json.loads((ROOT / "release.json").read_text())
    source = (ROOT / "imprint_nodes.py").read_text()

    project = pyproject["project"]
    comfy = pyproject["tool"]["comfy"]
    versions = {
        project["version"],
        release["version"],
        re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE).group(1),
    }
    if len(versions) != 1:
        raise ValueError(f"distribution versions differ: {sorted(versions)}")
    if project["name"] != release["package"]:
        raise ValueError("project and release package names differ")
    if project["urls"]["Repository"] != release["repository"]:
        raise ValueError("project and release repository URLs differ")
    if "cryptography>=42.0.0" not in project["dependencies"]:
        raise ValueError("prompt encryption requires the cryptography dependency")
    if release["registryStatus"] == "published":
        if comfy["PublisherId"] == "PUBLISHER_ID_REQUIRED":
            raise ValueError("published release cannot use publisher guard")
        if not release["registryUrl"]:
            raise ValueError("published release requires registryUrl")
    required = {"imprint_nodes.py", "__init__.py", "README.md", "LICENSE"}
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    if missing:
        raise ValueError(f"missing required files: {missing}")

    print(f"Validated {project['name']} {project['version']}")
    if release["registryStatus"] != "published":
        print("Registry publication is still pending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())