import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


def format_date(iso_date: str) -> str:
    """Convert ISO 8601 date string to readable format."""
    if not iso_date:
        return "N/A"
    try:
        return datetime.fromisoformat(iso_date.replace("Z", "")).strftime("%B %d, %Y")
    except ValueError:
        return iso_date


def review_to_markdown(review: Dict[str, Any]) -> str:
    """Convert single review to Markdown format."""
    return "\n".join(
        [
            f"### Review by {review.get('reviewer_name', 'Anonymous')} ({review.get('reviewer_location', 'Unknown')})",
            f"- **Posted on**: {format_date(review.get('review_date'))}",
            f"- **Experience Date**: {format_date(review.get('review_date_of_experience'))}",
            f"- **Rating**: {review.get('review_rating', 'N/A')}",
            f"- **Title**: *{review.get('review_title', '').strip()}*",
            f"\n{review.get('review_content', '').strip()}",
            f"\n[View Full Review]({review.get('review_url', '#')})",
            "\n---\n",
        ]
    )


def generate_markdown_content(reviews: List[Dict[str, Any]]) -> str:
    """Generate complete Markdown document from reviews."""
    if not reviews:
        return "# No reviews available."

    company = reviews[0]
    return "\n".join(
        [
            f"# {company.get('company_name', 'Unknown')} Review Summary",
            f"[Visit Website]({company.get('company_website', '#')})",
            (
                f"![Company Logo]({company['company_logo']})"
                if company.get("company_logo")
                else ""
            ),
            f"**Overall Rating**: {company.get('company_overall_rating', 'N/A')}",
            f"**Total Reviews**: {company.get('company_total_reviews', 'N/A')}",
            f"**Location**: {company.get('company_location', 'N/A')}",
            f"**Industry**: {', '.join(company.get('breadcrumbs', []))}",
            f"\n> {company.get('company_about', '').strip()}",
            "---\n",
            *[review_to_markdown(review) for review in reviews],
        ]
    )


def load_reviews(file_path: Path) -> List[Dict[str, Any]]:
    """Load reviews from JSON file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def save_markdown(content: str, output_path: Path):
    """Save content to Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            f.write(content)
        print(f"Markdown file created: {output_path}")
    except IOError as e:
        raise IOError(f"Error saving {output_path}: {e}")


def main():
    """Convert JSON reviews to Markdown format."""
    parser = argparse.ArgumentParser(
        description="Convert Trustpilot JSON reviews to Markdown"
    )
    parser.add_argument(
        "--input", default="trustpilot_reviews.json", help="Input JSON file path"
    )
    parser.add_argument(
        "--output", default="trustpilot_reviews.md", help="Output Markdown file path"
    )
    args = parser.parse_args()

    try:
        reviews = load_reviews(Path(args.input))
        markdown = generate_markdown_content(reviews)
        save_markdown(markdown, Path(args.output))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()