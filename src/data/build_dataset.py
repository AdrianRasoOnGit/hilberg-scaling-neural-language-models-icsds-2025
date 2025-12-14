
from pathlib import Path
from tqdm import tqdm

from .download import download_dump
from .extract import iterate_xml
from .clean import clean_wikitext
from .shard import ParquetSharder

def build_dataset(
    raw_dir="data/raw",
    processed_dir="data/processed",
    shard_size=50_000,
):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    print("·" * 25)
    print("Wikipedia Dataset Adquisition and Cleaning start.")
    print("·" * 25)

    # Download dump if not present
    dump_path = download_dump(output_dir=raw_dir)

    # Initialize sharder
    sharder = ParquetSharder(output_dir=processed_dir, shard_size=shard_size)

    # Extract, clean, and shard
    print("[build_dataset] Extracting and cleaning entries.")

    for page in tqdm(iterate_xml(dump_path), desc="Processing entries", unit="entry"):
        clean_text = clean_wikitext(page["text"])

        if not clean_text.strip():
            continue

        sharder.add({
            "id": page["id"],
            "title": page["title"],
            "text": clean_text,
        })

    # Final flush
    print("[build_dataset] Dataset built.")
    sharder.flush()

    print("·" * 25)
    print("Wikipedia Dataset Adquisition and Cleaning finished.")
    print("·" * 25)

if __name__ == "__main__":
    build_dataset()    
