import json
from pathlib import Path
from typing import Any, Dict


def load_groups(group_name: str) -> Dict[str, Any]:
    """Simple aux file to get the group info from json."""
    groups_file = Path(__file__).parent / f"{group_name}.json"

    if not groups_file.exists():
        raise FileNotFoundError(f"Group configuration file not found: {groups_file}. ")

    with open(f"{group_name}.json", "r") as file:
        return json.load(file)
