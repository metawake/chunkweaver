"""Quick inspection of HeadingDetector results."""
from heading_detector import detect_headings
from pathlib import Path

fpath = Path("10k_filings/microsoft_10k.txt")
text = fpath.read_text()
candidates = detect_headings(text, min_score=5.0)

print(f"=== {fpath.name}: {len(candidates)} headings at score>=5.0 ===\n")

for i, c in enumerate(candidates):
    if i < 30 or i % 80 == 0:
        sigs = ", ".join(c.signals)
        print(f"  L{c.line_number:5d} [{c.score:.1f}] {c.text[:75]:<75s} ({sigs})")

print(f"\n--- Signal distribution ---")
from collections import Counter
sig_counts = Counter()
for c in candidates:
    for s in c.signals:
        sig_counts[s] += 1
for sig, cnt in sig_counts.most_common():
    print(f"  {sig:20s}: {cnt:4d}")
