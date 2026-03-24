"""Download SEC 10-K filings and extract plain text."""

import html
import os
import re
import time
import urllib.request

FILINGS = {
    "apple": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019325000079/aapl-20250927.htm",
    "microsoft": "https://www.sec.gov/Archives/edgar/data/0000789019/000095017025100235/msft-20250630.htm",
    "tesla": "https://www.sec.gov/Archives/edgar/data/0001318605/000162828026003952/tsla-20251231.htm",
    "amazon": "https://www.sec.gov/Archives/edgar/data/0001018724/000101872426000004/amzn-20251231.htm",
    "jpmorgan": "https://www.sec.gov/Archives/edgar/data/0000019617/000162828026008131/jpm-20251231.htm",
    "jnj": "https://www.sec.gov/Archives/edgar/data/0000200406/000020040626000016/jnj-20251228.htm",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "10k_filings")
os.makedirs(OUT_DIR, exist_ok=True)


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</td>", "\t", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


for name, url in FILINGS.items():
    out_path = os.path.join(OUT_DIR, f"{name}_10k.txt")
    if os.path.exists(out_path):
        print(f"[skip] {name} already exists")
        continue

    print(f"[download] {name} ...")
    req = urllib.request.Request(url, headers={
        "User-Agent": "chunkweaver-research alex@example.com",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    text = html_to_text(raw)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  -> {len(text)} chars -> {out_path}")
    time.sleep(0.5)

print("\nDone.")
