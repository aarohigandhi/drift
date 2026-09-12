"""Write the training corpus: the Python language reference, as plain text.

Every Python install carries the language reference inside
``pydoc_data/topics.py``, as the prose behind ``help()``. It is real technical
English, about half a megabyte, and it is already on any machine that can run this
repository, so the corpus needs no download and cannot silently change under a URL.

The exact text depends on the Python version, so the version is recorded next to
the output. Results in the README were produced from Python 3.14.0.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pydoc_data.topics as topics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/corpus.txt", type=Path)
    args = ap.parse_args()

    text = "\n\n".join(topics.topics[k] for k in sorted(topics.topics))
    # Collapse to ASCII so the vocabulary does not grow a long tail of one-off
    # characters that appear a handful of times.
    text = text.encode("ascii", "ignore").decode("ascii")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" matters on Windows, where text mode would otherwise write \r\n and
    # hand the Java side a carriage return as an extra character.
    args.out.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    (args.out.parent / "corpus.meta").write_text(
        f"python={sys.version.split()[0]}\nchars={len(text)}\nvocab={len(set(text))}\nsha256_16={digest}\n",
        newline="\n",
    )
    print(f"{len(text):,} chars, vocab {len(set(text))}, sha256 {digest}, python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
