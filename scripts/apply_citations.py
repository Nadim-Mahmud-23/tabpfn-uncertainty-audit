#!/usr/bin/env python3
"""
Replace the inline [CIT: ...] stubs in paper/paper.tex with elsarticle-num
\\cite{key} commands, and replace the manual thebibliography block with
\\bibliography{refs}. Run ONLY after paper/refs.bib has been verified.

Mapping is by keyword -> bibkey; ambiguous bare "Romano et al. 2020" resolves to
the APS paper (romano2020aps); the equalized-coverage variant always carries the
"equalized"/"malice"/"Barber" markers and maps to romano2020eq.

Usage: python scripts/apply_citations.py        # dry run, prints unmatched stubs
       python scripts/apply_citations.py --write # apply in place
"""
from __future__ import annotations
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX = os.path.join(ROOT, "paper", "paper.tex")

# Ordered (specific first). Each: (regex tested against stub body, bibkey).
RULES = [
    (r"malice|equalized coverage|Barber, Sabatti", "romano2020eq"),
    (r"classification with valid and adaptive|Sesia", "romano2020aps"),
    (r"Romano et al\.?\\? ?2020", "romano2020aps"),   # bare -> APS
    (r"Mondrian|Vovk 2003", "vovk2003"),
    (r"Algorithmic Learning|Vovk et al\.?\\? ?2005|Vovk, Gammerman", "vovk2005"),
    (r"Papadopoulos", "papadopoulos2002"),
    (r"Angelopoulos", "angelopoulos2023"),
    (r"Sadinle", "sadinle2019"),
    (r"Lei", "lei2018"),
    (r"Gibbs.*Cherian|conditional guarantees|Gibbs et al\.?\\?\s*2023", "gibbs2023"),
    (r"Gibbs.*Cand.*2021|adaptive conformal", "gibbs2021"),
    (r"Jung|batch multivalid", "jung2023"),
    (r"Tibshirani|covariate shift", "tibshirani2019"),
    (r"Guo", "guo2017"),
    (r"Niculescu", "niculescu2005"),
    (r"Nixon|measuring calibration", "nixon2019"),
    (r"Grinsztajn|tree-based models", "grinsztajn2022"),
    (r"Gorishniy|revisiting", "gorishniy2021"),
    (r"McElfresh", "mcelfresh2023"),
    (r"M.ller|Transformers can do Bayesian", "pfn2022"),
    (r"TabPFN v2|Nature 2025|Accurate predictions on small data", "tabpfn2025"),
    (r"Hollmann|TabPFN.*2023|transformer that solves", "tabpfn2023"),
    (r"class-conditional|Ding et al\.?\\? ?2023", "ding2023"),
    (r"Retiring Adult|Ding.*2021|Ding, Hardt", "ding2021"),
]


def key_for(stub_body: str):
    for pat, key in RULES:
        if re.search(pat, stub_body, flags=re.I):
            return key
    return None


def main():
    src = open(TEX).read()
    # operate only on the inline citations (before \begin{thebibliography})
    head, sep, tail = src.partition(r"\begin{thebibliography}")
    unmatched = []

    # collapse runs of adjacent stubs into one \cite{a,b}
    def repl_run(m):
        body = m.group(0)
        stubs = re.findall(r"\[CIT:\s*(.*?)\]", body)
        keys = []
        for s in stubs:
            k = key_for(s)
            if k is None:
                unmatched.append(s)
            elif k not in keys:
                keys.append(k)
        return "\\cite{" + ",".join(keys) + "}" if keys else body

    new_head = re.sub(r"(\[CIT:[^\]]*\]\s*)+", repl_run, head)

    if unmatched:
        print("UNMATCHED stubs (fix RULES before --write):")
        for u in unmatched:
            print("   ", u)
    else:
        print("All inline stubs matched a bibkey.")

    # swap the manual bibliography for \bibliography{refs}
    if sep:
        new_tail = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                          r"\\bibliography{refs}", sep + tail, flags=re.S)
    else:
        new_tail = tail
    out = new_head + new_tail

    if "--write" in sys.argv and not unmatched:
        open(TEX, "w").write(out)
        print("Wrote", TEX)
    elif "--write" in sys.argv:
        print("NOT writing: unmatched stubs present.")
    else:
        print(f"Dry run. inline \\cite commands produced: {out[:len(new_head)].count(chr(92)+'cite')}")


if __name__ == "__main__":
    main()
