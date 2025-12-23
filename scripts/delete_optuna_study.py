#!/usr/bin/env python3
import argparse
import optuna


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete an Optuna study.")
    parser.add_argument("--study-name", required=True, help="Name of the study to delete.")
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_studies/hpo.db",
        help="Optuna storage URL (default: sqlite:///optuna_studies/hpo.db)",
    )
    args = parser.parse_args()

    storage = args.storage
    study_name = args.study_name

    print("Studies before deletion:")
    summaries = optuna.get_all_study_summaries(storage=storage)
    study_names = {s.study_name for s in summaries}
    for summary in summaries:
        print(f" - {summary.study_name}")

    if study_name not in study_names:
        print(f"\nStudy not found: {study_name}\n")
        return

    optuna.delete_study(study_name=study_name, storage=storage)
    print(f"\nDeleted study: {study_name}\n")

    print("Studies after deletion:")
    for summary in optuna.get_all_study_summaries(storage=storage):
        print(f" - {summary.study_name}")


if __name__ == "__main__":
    main()
