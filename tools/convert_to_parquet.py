"""
Converts the CSV files in the given directory to Parquet format.

Usage:
    python tools/convert_to_parquet.py -d data/01_raw

Notes:
    Created to just convert the initial raw data in csv format to parquet format
    for faster processing. This is not part of the Kedro pipeline.
"""


import argparse
import logging
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def convert_csv_to_parquet(directory: Path, n_jobs: int = -1):
    """
    Convert all CSV files in the given directory to Parquet format using parallel processing.

    Args:
        directory (Path): The path to the directory containing CSV files.
        n_jobs (int): The number of jobs to run in parallel. -1 means using all processors.
    """
    logger = logging.getLogger(__name__)

    csv_files = [file for file in directory.iterdir() if file.suffix == '.csv']
    tasks = [delayed(_convert_csv_to_parquet)(file, directory / file.with_suffix('.parquet').name) for file in csv_files]

    with tqdm(total=len(csv_files)) as pbar:
        for result in Parallel(n_jobs=n_jobs)(tasks):
            logger.info(result)
            pbar.update(1)

def _convert_csv_to_parquet(csv_path: Path, parquet_path: Path):
    """
    Convert a single CSV file to Parquet format.

    Args:
        csv_path (Path): Path to the CSV file.
        parquet_path (Path): Path for the output Parquet file.
    """
    try:
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        return f"Converted {csv_path} to {parquet_path}"
    except Exception as e:
        return f"Error converting {csv_path} to Parquet: {e}"

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to Parquet format.")
    parser.add_argument("-d", "--directory", type=str, default="data/01_raw", help="Directory to scan for CSV files (default: data/01_raw)")
    parser.add_argument("-j", "--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel (default: -1, meaning all processors)")
    args = parser.parse_args()

    directory_path = Path(args.directory)
    if not directory_path.is_dir():
        raise NotADirectoryError(f"{directory_path} is not a valid directory.")

    convert_csv_to_parquet(directory=directory_path, n_jobs=args.n_jobs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
