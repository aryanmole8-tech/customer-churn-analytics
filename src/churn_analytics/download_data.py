from pathlib import Path
from urllib.request import urlopen
import hashlib
from datetime import datetime, timezone

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Official IBM dataset URL
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
)


def download_dataset():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print("Dataset already exists. Skipping download.")
    else:
        print("Downloading IBM Telco Customer Churn dataset...")

        with urlopen(DATASET_URL, timeout=60) as response:
            data = response.read()

        if not data:
            raise ValueError("Downloaded dataset is empty.")

        OUTPUT_PATH.write_bytes(data)

        print("Dataset downloaded successfully!")

    # Record reproducibility information
    file_bytes = OUTPUT_PATH.read_bytes()
    checksum = hashlib.sha256(file_bytes).hexdigest()

    print("\n=== DATASET INFORMATION ===")
    print("Source:", DATASET_URL)
    print("File:", OUTPUT_PATH)
    print("File size:", len(file_bytes), "bytes")
    print("SHA256:", checksum)
    print(
        "Verified at:",
        datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    download_dataset()