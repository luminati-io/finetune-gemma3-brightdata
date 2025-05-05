import time
import json
import requests
from typing import Optional

# --- Configuration ---
API_KEY = "YOUR_API_KEY"  # Replace with your Bright Data API key
DATASET_ID = "gd_lm5zmhwd2sni130p"  # Replace with your Dataset ID
TARGET_URL = "https://www.trustpilot.com/review/hubspot.com"  # Target company page
OUTPUT_FILE = "trustpilot_reviews.json"  # Output file name
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
TIMEOUT = 30  # Request timeout in seconds


# --- Functions ---
def trigger_snapshot() -> Optional[str]:
    """Triggers a Bright Data snapshot collection job."""
    print(f"Triggering snapshot for: {TARGET_URL}")
    try:
        resp = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            headers=HEADERS,
            params={"dataset_id": DATASET_ID},
            json=[{"url": TARGET_URL}],
            timeout=TIMEOUT,
        )
        resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        snapshot_id = resp.json().get("snapshot_id")
        print(f"Snapshot triggered successfully. ID: {snapshot_id}")
        return snapshot_id
    except requests.RequestException as e:
        print(f"Error triggering snapshot: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding trigger response: {resp.text}")
    return None


def wait_for_snapshot(snapshot_id: str) -> Optional[list]:
    """Polls the API until snapshot data is ready and returns it."""
    check_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
    print(f"Waiting for snapshot {snapshot_id} to complete...")
    while True:
        try:
            resp = requests.get(
                check_url,
                headers=HEADERS,
                params={"format": "json"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            # Check if response is the final data (list) or status info (dict)
            if isinstance(resp.json(), list):
                print("Snapshot data is ready.")
                return resp.json()
            else:
                pass
        except requests.RequestException as e:
            print(f"Error checking snapshot status: {e}")
            return None  # Stop polling on error
        except json.JSONDecodeError:
            print(f"Error decoding snapshot status response: {resp.text}")
            return None  # Stop polling on error

        print("Data not ready yet. Waiting 30 seconds...")
        time.sleep(30)


def save_reviews(reviews: list, output_file: str) -> bool:
    """Saves the collected reviews list to a JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(reviews)} reviews to {output_file}")
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving reviews to file: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return False


def main():
    """Main execution flow for collecting Trustpilot reviews."""
    print("Starting Trustpilot review collection process...")
    snapshot_id = trigger_snapshot()
    if not snapshot_id:
        print("Failed to trigger snapshot. Exiting.")
        return

    reviews = wait_for_snapshot(snapshot_id)
    if not reviews:
        print("Failed to retrieve reviews from snapshot. Exiting.")
        return

    if not save_reviews(reviews, OUTPUT_FILE):
        print("Failed to save the collected reviews.")
    else:
        print("Review collection process completed.")


if __name__ == "__main__":
    main()