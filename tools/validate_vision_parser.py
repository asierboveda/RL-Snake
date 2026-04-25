import argparse
import json
from pathlib import Path

from vision_validation import evaluate_vision_parser, validation_report_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VisionParser against labeled dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/vision_captures/manifest.json"),
        help="Path to dataset manifest.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/vision_validation_report.md"),
        help="Path to write markdown report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/vision_captures/vision_validation_report.json"),
        help="Path to write machine-readable report.",
    )
    parser.add_argument(
        "--min-hard-cases",
        type=int,
        default=20,
        help="Minimum number of difficult cases to include.",
    )
    args = parser.parse_args()

    report = evaluate_vision_parser(
        manifest_path=args.manifest,
        min_hard_cases=args.min_hard_cases,
    )
    markdown = validation_report_markdown(report)
    json_payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    args.output_json.write_text(json_payload, encoding="utf-8")

    print(f"markdown report: {args.output_markdown}")
    print(f"json report: {args.output_json}")


if __name__ == "__main__":
    main()
