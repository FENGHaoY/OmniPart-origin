import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def resolve_default_paths() -> Dict[str, Path]:
    """
    Resolve default input (data_root) and output (index_path) paths.

    By default we expect the following structure (relative to the OmniPart repo root):

        ../project/data/            # raw Singapo PartNet-Mobility data
        ../project/processed_data/  # processed outputs (will be created if missing)
    """
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    data_root = project_root / "data"
    processed_root = project_root / "processed_data"
    index_path = processed_root / "dataset_index_all.json"
    return {
        "data_root": data_root,
        "processed_root": processed_root,
        "index_path": index_path,
    }


def scan_model_dir(category: str, model_dir: Path) -> Dict:
    """
    Scan a single <category>/<model_id> directory and return a record.
    """
    model_id = model_dir.name
    object_dir = model_dir
    object_json = object_dir / "object.json"

    record = {
        "category": category,
        "model_id": model_id,
        "object_dir": str(object_dir.resolve()),
        "object_json": str(object_json.resolve()),
        "objs_dir": str((object_dir / "objs").resolve()),
        "plys_dir": str((object_dir / "plys").resolve()),
        "imgs_dir": str((object_dir / "imgs").resolve()),
        "textures_dir": str((object_dir / "textures").resolve()),
    }

    return record


def scan_dataset(
    data_root: Path,
    allowed_categories: Sequence[str] | None = None,
) -> List[Dict]:
    """
    Recursively scan the dataset rooted at data_root and return a list of records.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    records: List[Dict] = []

    # Each immediate subdirectory under data_root is a category
    for category_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        category = category_dir.name

        # If allowed_categories is given, only keep those.
        if allowed_categories is not None and category not in allowed_categories:
            continue

        # Each immediate subdirectory under <category> is a model_id
        for model_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            # Require object.json to exist; skip otherwise to avoid malformed entries
            object_json = model_dir / "object.json"
            if not object_json.is_file():
                # Silent skip to keep the index clean; print to stderr for debugging.
                print(
                    f"[scan_dataset] Skip {model_dir} (missing object.json)",
                    file=sys.stderr,
                )
                continue

            record = scan_model_dir(category, model_dir)
            records.append(record)

    return records


def write_index(records: List[Dict], index_path: Path) -> None:
    """
    Write the dataset index to a JSON file.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} entries to {index_path}")


def parse_args() -> argparse.Namespace:
    defaults = resolve_default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Scan ../project/data and generate dataset_index_all.json\n\n"
            "Default locations (relative to the OmniPart repo root):\n"
            "  data_root       -> ../project/data\n"
            "  output index    -> ../project/processed_data/dataset_index_all.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(defaults["data_root"]),
        help="Path to the dataset root directory (default: ../project/data relative to repo root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(defaults["index_path"]),
        help="Path to the output JSON index file "
        "(default: ../project/processed_data/dataset_index_all.json).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    index_path = Path(args.output).expanduser().resolve()

    # For this research project we focus on the seven articulated categories.
    allowed_categories = [
        "Dishwasher",
        "Microwave",
        "Oven",
        "Refrigerator",
        "StorageFurniture",
        "Table",
        "WashingMachine",
    ]

    records = scan_dataset(data_root, allowed_categories=allowed_categories)
    write_index(records, index_path)


if __name__ == "__main__":
    main()

