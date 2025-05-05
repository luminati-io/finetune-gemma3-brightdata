import argparse
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Generator

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert at transforming customer reviews into insightful question-answer pairs. For each review, generate exactly 1 high-quality QA pair.

PURPOSE:
These QA pairs will train a customer service AI to understand feedback patterns about HubSpot products and identify actionable insights.

GUIDELINES FOR QUESTIONS:
- Make questions general and applicable to similar situations (not specific to the individual review)
- Phrase questions from a business stakeholder perspective (e.g., "What feature gaps are causing customer frustration?")
- Focus on extracting insights about product features, pricing, usability, customer service, or business impact
- Questions should be clear, concise, and directly relevant to improving HubSpot's products or services

GUIDELINES FOR ANSWERS:
- Provide substantive, analytical responses (3-5 sentences)
- Include specific insights from the review without directly quoting it
- Focus on actionable recommendations when possible
- Maintain objectivity - acknowledge both strengths and weaknesses if present
- Structure answers to be useful for product teams, customer success, or executive stakeholders

FORMAT REQUIREMENTS:
- Q: A clear, business-focused question that could apply to multiple similar reviews
- A: A thoughtful analysis that extracts key insights while providing actionable context
- Do NOT use any special characters or formatting like **, \n, #, *, _, or any other Markdown syntax
- Use plain text only without any formatting or special characters
- Do not include bullet points, numbered lists, or any other structural elements
"""


class QAGenerator:
    def __init__(self, model: str = "gpt-4o", max_workers: int = 4):
        self.client = self._create_openai_client()
        self.model = model
        self.max_workers = max_workers
        self.total_tokens = 0
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "empty": 0,
            "invalid_format": 0,
        }

    def _create_openai_client(self) -> OpenAI:
        """Initialize and return OpenAI client."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=(
            retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(RateLimitError)
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry #{retry_state.attempt_number} for review"
        ),
    )
    def _generate_qa(self, review_text: str) -> Optional[Dict[str, str]]:
        """Generate a single QA pair from review text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": review_text},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            self.total_tokens += response.usage.total_tokens
            content = response.choices[0].message.content.strip()

            # Strict format validation
            if not content.startswith("Q: ") or "A: " not in content:
                self.stats["invalid_format"] += 1
                return None

            question = content.split("A:")[0][3:].strip()
            answer = content.split("A:")[1].strip()

            if not question or not answer:
                self.stats["empty"] += 1
                return None

            return {"id": str(uuid.uuid4()), "question": question, "answer": answer}

        except Exception as e:
            logger.debug(f"Error processing review: {str(e)}")
            self.stats["failed"] += 1
            return None

    def process_reviews(
        self,
        input_path: Path,
        output_path: Path,
        batch_size: int = 50,
        save_every: int = 100,
    ) -> None:
        """Process reviews in parallel and generate QA pairs."""
        logger.info(f"Starting QA generation with {self.max_workers} workers")

        # Initialize output file
        if not output_path.exists():
            output_path.write_text("[]")

        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            batch = []

            for review in self._read_reviews(input_path):
                self.stats["processed"] += 1
                futures.append(executor.submit(self._generate_qa, review["review"]))

                # Process completed futures
                if len(futures) >= batch_size:
                    batch = self._process_completed_futures(
                        futures, batch, output_path, save_every
                    )
                    futures = []

            # Process remaining futures
            if futures:
                self._process_completed_futures(futures, batch, output_path, save_every)

        logger.info(
            f"Completed processing {self.stats['processed']} reviews\n"
            f"Successful QA pairs: {self.stats['success']}\n"
            f"Token usage: {self.total_tokens}\n"
            f"Failures: {self.stats['failed']} (Empty: {self.stats['empty']}, "
            f"Invalid format: {self.stats['invalid_format']})"
        )

    def _process_completed_futures(
        self, futures: list, batch: list, output_path: Path, save_every: int
    ) -> list:
        """Process completed futures and save progress."""
        new_batch = batch.copy()

        for future in as_completed(futures):
            result = future.result()
            if result:
                new_batch.append(result)
                self.stats["success"] += 1

                # Save progress periodically
                if len(new_batch) >= save_every:
                    self._save_batch(new_batch, output_path)
                    logger.info(
                        f"Saved {len(new_batch)} QA pairs (Total: {self.stats['success']})"
                    )
                    new_batch = []

        return new_batch

    def _read_reviews(self, input_path: Path) -> Generator[Dict, None, None]:
        """Lazily read reviews from JSONL file."""
        try:
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Error reading {input_path}: {str(e)}")
            raise

    def _save_batch(self, batch: List[Dict], output_path: Path) -> None:
        """Atomically append batch to output file."""
        try:
            # Read existing data
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []

            # Append new data
            existing.extend(batch)

            # Atomic write
            temp_path = output_path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

            temp_path.replace(output_path)

        except Exception as e:
            logger.error(f"Error saving batch: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from customer reviews in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="trustpilot_reviews_chunks.jsonl",
        help="Input JSONL file containing reviews",
    )
    parser.add_argument(
        "--output",
        default="trustpilot_qa_pairs.json",
        help="Output JSON file for QA pairs",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model to use for generation"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers for processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of reviews to process in each batch",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save progress after every N successful generations",
    )
    args = parser.parse_args()

    try:
        generator = QAGenerator(model=args.model, max_workers=args.max_workers)
        generator.process_reviews(
            Path(args.input),
            Path(args.output),
            batch_size=args.batch_size,
            save_every=args.save_every,
        )
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
