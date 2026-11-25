"""Install the PAL modelling skill into an agent's skills directory.

For example: python scripts/install_pal_skill.py --target .agents/skills
The target is a directory of skills, not the skill's own directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def install_skill(target: Path, overwrite: bool = False) -> Path:
    """Copy the versioned PAL skill to a chosen agent skills directory."""
    source = Path(__file__).resolve().parents[1] / "skills" / "pal-modeling"
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError("PAL skill source was not found")
    destination = target.expanduser() / source.name
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists. Use --force to replace it.")
        if not destination.is_dir() or destination.is_symlink():
            raise ValueError(f"Refusing to replace a non-directory or symlink: {destination}")
        shutil.rmtree(destination)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def main() -> None:
    """Copy the skill without requiring network access or an agent-specific CLI."""
    parser = argparse.ArgumentParser(description="Install the PAL skill in an agent skills directory.")
    parser.add_argument("--target", type=Path, default=Path(".agents/skills"), help="Parent skills directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing PAL skill directory")
    arguments = parser.parse_args()
    print(f"Installed PAL modelling skill at {install_skill(arguments.target, overwrite=arguments.force)}")


if __name__ == "__main__":
    main()
