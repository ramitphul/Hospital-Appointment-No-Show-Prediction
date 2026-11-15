"""
Master execution script for complete end-to-end Machine Learning Pipeline.
Executes data loading, validation, EDA generation, multi-model training,
hyperparameter tuning, probability calibration, threshold optimization,
SHAP explainability, and artifact serialization.

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --scenario booking --no-tune
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

# Configure Matplotlib backend and cache
os.environ["MPLCONFIGDIR"] = str(root_dir / "artifacts" / ".matplotlib")

from src.data.eda import generate_eda_visualizations, save_eda_report
from src.data.load_data import load_dataset
from src.data.validate_data import save_validation_report, validate_dataset
from src.models.train import run_full_training_pipeline
from src.utils.config import get_resolved_path, load_config
from src.utils.logger import get_logger

logger = get_logger("train_pipeline_script")


def main():
    parser = argparse.ArgumentParser(description="Execute Hospital No-Show End-to-End ML Pipeline.")
    parser.add_argument("--scenario", type=str, default="booking", choices=["booking", "pre_appointment"],
                        help="Prediction scenario: 'booking' (default, leak-free) or 'pre_appointment'.")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search for faster execution.")
    parser.add_argument("--tune-iter", type=int, default=8, help="Number of iterations for hyperparameter tuning.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA chart generation if already produced.")
    args = parser.parse_args()

    t_start = time.time()
    config = load_config()

    print("\n" + "=" * 70)
    print("   HOSPITAL APPOINTMENT NO-SHOW PREDICTION - ML TRAINING PIPELINE   ")
    print("=" * 70)
    print(f"Scenario:             {args.scenario}")
    print(f"Hyperparameter Tuning:{'Disabled' if args.no_tune else f'Enabled ({args.tune_iter} iterations)'}")
    print(f"Random Seed:          {config['project']['random_seed']}")
    print("-" * 70)

    # Step 1: Load and Validate Data
    logger.info("Step 1/5: Loading and validating dataset...")
    df, data_source = load_dataset()
    val_report = validate_dataset(df)
    metrics_dir = get_resolved_path(config["paths"]["metrics_dir"])
    save_validation_report(val_report, metrics_dir / "data_validation_report.json")
    print(f"[✓] Data Loaded: {len(df):,} records ({data_source})")
    print(f"[✓] Validation Status: {val_report.validation_status} | Baseline No-Show Rate: {val_report.target_no_show_rate:.2%}")

    # Step 2: Exploratory Data Analysis
    if not args.skip_eda:
        logger.info("Step 2/5: Generating exploratory data analysis and charts...")
        eda_dir = get_resolved_path(config["paths"]["eda_dir"])
        stats = generate_eda_visualizations(df, eda_dir)
        save_eda_report(stats, eda_dir)
        print(f"[✓] EDA complete. Visualizations saved to {eda_dir}")
    else:
        print("[*] Skipping EDA as requested.")

    # Step 3: Execute Training & Evaluation Pipeline
    logger.info("Step 3/5: Training 6 ML algorithms, hyperparameter tuning, and probability calibration...")
    metadata = run_full_training_pipeline(
        df=df,
        scenario=args.scenario,
        tune_top_model=(not args.no_tune),
        n_tune_iter=args.tune_iter,
    )

    # Step 4: Summary of Results
    test_m = metadata["test_metrics"]
    duration = time.time() - t_start

    print("\n" + "=" * 70)
    print("                      TRAINING PIPELINE COMPLETE                     ")
    print("=" * 70)
    print(f"Best Performing Algorithm:  {metadata['best_algorithm']}")
    print(f"Optimal Decision Threshold: {metadata['optimal_threshold']:.3f} (default: 0.50)")
    print(f"Engineered Features Count:  {metadata['feature_count']}")
    print(f"Calibrated Brier Score:     {metadata['calibrated_brier_score']:.4f}")
    print("-" * 70)
    print("HELD-OUT TEST SET EVALUATION METRICS:")
    print(f"  • ROC-AUC Score:          {test_m['roc_auc']:.4f}")
    print(f"  • PR-AUC (Avg Precision): {test_m['pr_auc']:.4f}")
    print(f"  • Recall (Sensitivity):   {test_m['recall']:.4f} ({test_m['recall']*100:.1f}% of no-shows identified)")
    print(f"  • Precision:              {test_m['precision']:.4f}")
    print(f"  • Specificity:            {test_m['specificity']:.4f}")
    print(f"  • F1-Score:               {test_m['f1_score']:.4f}")
    print(f"  • Overall Accuracy:       {test_m['accuracy']:.4f}")
    print("-" * 70)
    print(f"Total Execution Time:       {duration:.1f} seconds")
    print(f"Serialized Model Artifacts: artifacts/models/")
    print(f"Evaluation Metrics & Plots: artifacts/metrics/")
    print(f"Explainability Plots:       artifacts/shap/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
