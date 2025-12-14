
import requests
import os
from pathlib import Path
from tqdm import tqdm

dump_url = (
    "https://dumps.wikimedia.org/enwiki/20250901/enwiki-20250901-pages-articles-multistream.xml.bz2"
)

output_dir = Path("data/raw")
filename = "wikipedia_dump.xml.bz2"

def download_dump(
    url: str = dump_url,
    output_dir: Path = output_dir,
    filename: str = filename,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[download.py] File already exists: {output_path}")
        return output_path

    print(f"[download.py] Wikipedia dump download started.")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024

    with open(output_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"[download.py] Download complete in {output_path}")
    return output_path
