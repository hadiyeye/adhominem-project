import csv, random
from pathlib import Path

random.seed(42)

inp = Path("/workspace/AdHominem/data_0219/amazon_test_rewrite.tsv")
out = Path("/workspace/AdHominem/data_0219/amazon_test_rewrite_shuflabel.tsv")

rows = []
with inp.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

labels = [r["sentiment"] for r in rows]
random.shuffle(labels)

for r, y in zip(rows, labels):
    r["sentiment"] = y

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id","sentiment","review"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print("[ok] wrote:", out)
print("[info] n=", len(rows))
