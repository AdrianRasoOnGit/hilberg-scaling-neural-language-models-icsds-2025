
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetSharder:
    def __init__(self, output_dir="data/processed", shard_size=50000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.current_rows = []
        self.shard_index = 0

        print(f"[shard.py] Sharded started to {output_dir}")

    def _write_shard(self):
        if not self.current_rows:
            return

        shard_name = f"shard_{self.shard_index:03d}.parquet"
        shard_path = self.output_dir / shard_name

        print(f"[shard.py] Writing {len(self.current_rows)} rows to {shard_path}")
        table = pa.Table.from_pylist(self.current_rows)
        pq.write_table(table, shard_path)

        self.shard_index += 1
        self.current_rows = []

    def add(self, page_dict):
        self.current_rows.append(page_dict)

        if len(self.current_rows) >= self.shard_size:
            self._write_shard()

    def flush(self):
        self._write_shard()
        print("[shard.py] Flush complete.")
