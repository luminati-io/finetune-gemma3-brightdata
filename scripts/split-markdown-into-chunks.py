import os
import argparse
import json
import uuid
import logging
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_openai_client() -> OpenAI:
    """Create and return OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def create_text_splitter(
    chunk_size: int = 1024, chunk_overlap: int = 256
) -> RecursiveCharacterTextSplitter:
    """Create text splitter with specified parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def improve_review_chunk(text: str, client: OpenAI, model: str = "gpt-4o") -> str:
    """Improve coherence of a review chunk."""
    prompt = """Improve this review's clarity while preserving its meaning:
{text}

Return only the improved text without additional commentary."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error improving chunk: {e}")
        return text


def process_chunks_parallel(
    chunks: List[str], client: OpenAI, max_workers: int = 4, model: str = "gpt-4o"
) -> List[str]:
    """Process chunks in parallel."""
    logger.info(f"Processing {len(chunks)} chunks with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(lambda c: improve_review_chunk(c, client, model), chunks)
        )


def process_markdown_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    improve_coherence: bool = True,
    max_workers: int = 4,
    model: str = "gpt-4o",
):
    """Process markdown file into improved chunks."""
    logger.info(f"Processing {input_path}")
    try:
        text = input_path.read_text(encoding="utf-8")
        splitter = create_text_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)

        if improve_coherence:
            client = create_openai_client()
            chunks = process_chunks_parallel(chunks, client, max_workers, model)

        chunks_with_ids = [
            {"id": str(uuid.uuid4()), "review": chunk.strip()} for chunk in chunks
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks_with_ids:
                f.write(json.dumps(chunk) + "\n")

        logger.info(f"Saved {len(chunks_with_ids)} chunks to {output_path}")

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Split reviews into chunks")
    parser.add_argument(
        "--input", default="trustpilot_reviews.md", help="Input markdown file"
    )
    parser.add_argument(
        "--output", default="trustpilot_reviews_chunks.jsonl", help="Output JSONL file"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=256, help="Chunk overlap in characters"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Max parallel workers"
    )
    args = parser.parse_args()

    process_markdown_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
