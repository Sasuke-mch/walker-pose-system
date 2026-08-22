import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a COCO image subset using hard links "
            "without duplicating image data."
        )
    )
    parser.add_argument(
        "--source-images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not args.source_images.is_dir():
        raise FileNotFoundError(
            f"Source image directory not found: "
            f"{args.source_images}"
        )

    if not args.annotation.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: "
            f"{args.annotation}"
        )

    with args.annotation.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        coco = json.load(file)

    image_records = coco.get("images", [])

    if not isinstance(image_records, list):
        raise TypeError(
            "COCO annotation does not contain an images list"
        )

    expected_names = [
        record["file_name"]
        for record in image_records
    ]

    if len(expected_names) != len(set(expected_names)):
        raise ValueError(
            "Duplicate image filenames found in annotation"
        )

    args.output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    created = 0
    existing = 0
    missing = []
    failures = []

    for index, file_name in enumerate(
        expected_names,
        start=1,
    ):
        source = args.source_images / file_name
        destination = args.output_images / file_name

        if not source.is_file():
            missing.append(file_name)
            continue

        if destination.exists():
            existing += 1
            continue

        try:
            os.link(source, destination)
            created += 1
        except OSError as error:
            failures.append(
                {
                    "file_name": file_name,
                    "error": str(error),
                }
            )

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(expected_names)}"
            )

    actual_files = {
        path.name
        for path in args.output_images.iterdir()
        if path.is_file()
    }

    expected_set = set(expected_names)

    output_missing = sorted(
        expected_set - actual_files
    )

    output_extra = sorted(
        actual_files - expected_set
    )

    manifest = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "operation": "create_hard_link_image_view",
        "source_images": str(
            args.source_images.resolve()
        ),
        "annotation": str(
            args.annotation.resolve()
        ),
        "output_images": str(
            args.output_images.resolve()
        ),
        "expected_images": len(expected_names),
        "created_hard_links": created,
        "already_existing": existing,
        "source_missing": missing,
        "hard_link_failures": failures,
        "actual_output_files": len(actual_files),
        "output_missing": output_missing,
        "output_extra": output_extra,
    }

    args.manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.manifest.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Expected images:", len(expected_names))
    print("Created hard links:", created)
    print("Already existing:", existing)
    print("Source missing:", len(missing))
    print("Hard-link failures:", len(failures))
    print("Actual output files:", len(actual_files))
    print("Output missing:", len(output_missing))
    print("Output extra:", len(output_extra))
    print("Manifest:", args.manifest)

    if missing:
        print()
        print("First missing source images:")
        for name in missing[:20]:
            print(" ", name)

    if failures:
        print()
        print("First hard-link failures:")
        for item in failures[:20]:
            print(
                " ",
                item["file_name"],
                item["error"],
            )

    if (
        missing
        or failures
        or output_missing
        or output_extra
    ):
        raise SystemExit(
            "COCO IMAGE VIEW BUILD FAILED"
        )

    print()
    print("COCO IMAGE VIEW BUILD PASSED")


if __name__ == "__main__":
    main()
