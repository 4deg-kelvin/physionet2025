#!/usr/bin/env python
"""Mechanical prose check for main.tex: the tics that make a draft read as machine-written.

    python paper/prose_check.py [paper/main.tex]

Reports, per section: paragraph lengths (a flat profile is a tic), sentence-length spread,
sentence openers that repeat, stock phrases, stacked parentheticals, "lists of three" per
paragraph, and hedges per 1,000 words.  Thresholds are advisory; the point is to see the
distribution, not to hit a number.  Macros (\\foo) are counted as one word.
"""
from __future__ import annotations

import collections
import re
import statistics
import sys
from pathlib import Path

STOCK = [
    "it is worth noting", "note that", "notably", "importantly", "crucially", "interestingly",
    "in other words", "that is to say", "it is important to", "it should be noted", "delve",
    "robust", "leverage", "underscore", "highlight", "showcase", "pivotal", "landscape",
    "in summary", "overall,", "in conclusion", "furthermore", "moreover", "additionally",
    "which is consistent with", "consistent with", "as expected", "taken together", "to this end",
    "serves as", "plays a role", "sheds light", "paves the way", "a testament", "multifaceted",
    "comprehensive", "seamless", "nuanced", "holistic",
]
HEDGES = ["may", "might", "could", "suggests", "suggest", "likely", "plausible", "possibly", "appears", "seems", "tends"]


def strip_tex(s: str) -> str:
    s = re.sub(r"(?m)^%.*$", "", s)
    s = re.sub(r"\\(cite[pt]?|ref|label|input|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " REF ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", "\n", s)
    s = re.sub(r"\$[^$]*\$", " NUM ", s)
    s = re.sub(r"\\[A-Za-z]+\*?(\{\})?", " MACRO ", s)
    s = re.sub(r"[{}~]", " ", s)
    return s


def sections(tex: str):
    parts = re.split(r"\\(?:sub)*section\*?\{([^}]*)\}", tex)
    yield "front", parts[0]
    for i in range(1, len(parts), 2):
        yield parts[i], parts[i + 1]


def paragraphs(block: str):
    for para in re.split(r"\n\s*\n", block):
        p = " ".join(para.split())
        if len(p.split()) >= 25 and not p.startswith(("\\item", "&")):
            yield p


def sentences(p: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", p) if len(s.split()) >= 3]


def main(path: str) -> None:
    tex = Path(path).read_text()
    body = tex[tex.index("\\begin{abstract}"):] if "\\begin{abstract}" in tex else tex
    body = body.split("\\ack{")[0]
    total_words = 0
    stock_hits: collections.Counter = collections.Counter()
    hedge_hits = 0
    print(f"{'section':42s} {'paras':>5s} {'words/para':>22s} {'sent len mean/sd':>17s} {'rep.openers':>11s} {'parens':>6s} {'triads':>6s}")
    for title, block in sections(body):
        clean = strip_tex(block)
        paras = list(paragraphs(clean))
        if not paras:
            continue
        plens = [len(p.split()) for p in paras]
        sents = [s for p in paras for s in sentences(p)]
        slens = [len(s.split()) for s in sents] or [0]
        openers = collections.Counter(" ".join(s.split()[:2]).lower() for s in sents)
        rep = sum(c - 1 for c in openers.values() if c > 1)
        parens = sum(p.count("(") for p in paras)
        triads = sum(len(re.findall(r"\b\w+, \w+(?: \w+)?,? and \w+", p)) for p in paras)
        total_words += sum(plens)
        low = " ".join(paras).lower()
        for ph in STOCK:
            n = low.count(ph)
            if n:
                stock_hits[ph] += n
        hedge_hits += sum(len(re.findall(rf"\b{h}\b", low)) for h in HEDGES)
        sd = statistics.pstdev(plens) if len(plens) > 1 else 0.0
        ssd = statistics.pstdev(slens) if len(slens) > 1 else 0.0
        print(f"{title[:42]:42s} {len(paras):5d} {min(plens):4d}-{max(plens):<4d} (sd {sd:5.1f}) {statistics.mean(slens):7.1f}/{ssd:<6.1f}  {rep:11d} {parens:6d} {triads:6d}")
    print(f"\nwords (body, prose paragraphs only): {total_words}")
    print(f"hedges per 1,000 words: {1000 * hedge_hits / max(total_words, 1):.1f}  (academic prose is typically 5-12)")
    print("stock phrases:", dict(stock_hits.most_common()) or "none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("main.tex")))
