import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DatasetUploader:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.api = HfApi()
        self.stats = {"start_time": time.time(), "processed": 0, "skipped": 0}

    def validate_dataset(self, data: List[Dict]) -> bool:
        """Validate dataset structure and content."""
        required_keys = {"id", "question", "answer"}
        for item in data:
            if not all(key in item for key in required_keys):
                logger.error(
                    f"Missing required keys in item: {item.get('id', 'unknown')}"
                )
                return False
            if not isinstance(item["question"], str) or not item["question"].strip():
                logger.error(f"Invalid question in item: {item['id']}")
                return False
            if not isinstance(item["answer"], str) or not item["answer"].strip():
                logger.error(f"Invalid answer in item: {item['id']}")
                return False
        return True

    def create_hf_dataset(self, data: List[Dict]) -> Dataset:
        """Create Hugging Face dataset with train/test split."""
        try:
            # Create 90/10 train-test split
            split_index = int(len(data) * 0.9)
            train_data = data[:split_index]
            test_data = data[split_index:]

            return DatasetDict(
                {
                    "train": Dataset.from_list(train_data),
                    "test": Dataset.from_list(test_data),
                }
            )
        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}")
            raise

    def upload_dataset(
        self,
        dataset: DatasetDict,
        repo_name: str,
        private: bool = False,
        retries: int = 3,
    ) -> bool:
        """Upload dataset to Hub with retries and progress tracking."""
        for attempt in range(retries):
            try:
                logger.info(f"Upload attempt {attempt + 1}/{retries}")

                # Login for each attempt in case token expires
                login(token=self.hf_token)

                # Push to Hub with progress
                dataset.push_to_hub(
                    repo_id=repo_name,
                    private=private,
                    commit_message=f"Add dataset v{int(time.time())}",
                    max_shard_size="500MB",
                )

                return True

            except HfHubHTTPError as e:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                logger.warning(
                    f"Upload failed (attempt {attempt + 1}): {str(e)}\n"
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error during upload: {str(e)}")
                break

        return False

    def process_file(
        self, input_path: Path, repo_name: str, private: bool = False
    ) -> bool:
        """Process input file and upload to Hugging Face Hub."""
        try:
            # Load and validate data
            logger.info(f"Loading dataset from {input_path}")
            with input_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error("Dataset must be a list of QA pairs")
                return False

            self.stats["processed"] = len(data)

            if not self.validate_dataset(data):
                logger.error("Dataset validation failed")
                return False

            # Create HF dataset
            logger.info("Creating Hugging Face dataset")
            dataset = self.create_hf_dataset(data)

            # Upload to Hub
            logger.info(f"Uploading to {repo_name}")
            if not self.upload_dataset(dataset, repo_name, private):
                logger.error("All upload attempts failed")
                return False

            # Report stats
            elapsed = time.time() - self.stats["start_time"]
            logger.info(
                f"Successfully uploaded {self.stats['processed']} items\n"
                f"Train samples: {len(dataset['train'])}\n"
                f"Test samples: {len(dataset['test'])}\n"
                f"Elapsed time: {elapsed:.2f} seconds"
            )
            return True

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload QA dataset to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="trustpilot_qa_pairs.json",
        help="Path to input JSON file containing QA pairs",
    )
    parser.add_argument(
        "--repo",
        default="triposatt/trustpilot-reviews-qa-dataset",
        help="HF repository name (format: username/dataset-name)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    args = parser.parse_args()

    # Load HF token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        sys.exit(1)

    # Process and upload
    uploader = DatasetUploader(hf_token)
    success = uploader.process_file(Path(args.input), args.repo, args.private)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
