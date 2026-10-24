
import json
from pathlib import Path
from datetime import datetime

path = Path("backend/data/processed_orderbook.jsonl")

count = 0
min_time = None
max_time = None
symbols = set()

with path.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        record = json.loads(line)
        timestamp = datetime.fromisoformat(
            record["timestamp"].replace("Z", "+00:00")
        )

        count += 1
        symbols.add(record.get("symbol", "UNKNOWN"))

        if min_time is None or timestamp < min_time:
            min_time = timestamp

        if max_time is None or timestamp > max_time:
            max_time = timestamp

print("Total records:", count)
print("First timestamp:", min_time)
print("Last timestamp:", max_time)
print("Symbols:", sorted(symbols))
