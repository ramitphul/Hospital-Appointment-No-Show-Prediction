"""
Command-line script for dataset acquisition and verification.
Usage:
    python scripts/download_data.py
    python scripts/download_data.py --synthetic --samples 10000
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.data.load_data import create_sample_dataset, generate_synthetic_data, load_dataset
from src.data.validate_data import save_validation_report, validate_dataset
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("download_data_script")


def main():
    parser = argparse.ArgumentParser(description="Download and validate Brazilian Medical Appointment dataset.")
    parser.add_argument("--synthetic", action="store_true", help="Force generation of synthetic demonstration data.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples if --synthetic is set.")
    parser.add_argument("--sample-slice", type=int, default=300, help="Number of records for quick testing sample CSV.")
    args = parser.parse_args()

    config = load_config()
    raw_path = get_resolved_path(config["paths"]["raw_data"])
    metrics_dir = get_resolved_path(config["paths"]["metrics_dir"])

    if args.synthetic:
        logger.info(f"Generating synthetic demonstration data ({args.samples} samples)...")
        df = generate_synthetic_data(num_samples=args.samples)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        data_source = "Synthetic Demonstration Dataset"
    else:
        df, data_source = load_dataset(allow_download=True, fallback_synthetic=True)

    print(f"\n==========================================")
    print(f"Dataset Acquired: {len(df):,} records")
    print(f"Source: {data_source}")
    print(f"Saved to: {raw_path}")
    print(f"==========================================\n")

    # Validate dataset
    report = validate_dataset(df)
    report_path = metrics_dir / "data_validation_report.json"
    save_validation_report(report, report_path)

    # Create sample slice
    sample_path = get_resolved_path(config["paths"]["sample_data"])
    create_sample_dataset(df, sample_size=args.sample_slice, output_path=sample_path)
    print(f"Sample test CSV created at: {sample_path}\n")


if __name__ == "__main__":
    main()
