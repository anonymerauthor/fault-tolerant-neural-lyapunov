"""Aggregate fault-bank verification chunks (abcrown logs) into a per-mode table.

A variant counts as certified if every region of its chunk is 'safe'. For chunks that are not
fully safe, the variants are reported as 'unresolved' unless finer chunks (smaller ranges)
inside the same directory certify them (the finest available evidence wins).

Usage: python src/summarize_faults.py runs/f2b_grid/verify_faults_rho0.165_k0.005
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def chunk_status(log: Path):
    txt = log.read_text(errors="ignore")
    res = re.findall(r"Result: (\w+) in ([\d.]+) seconds", txt)
    return [r[0] for r in res], sum(float(r[1]) for r in res)


def main(vdir: str):
    vdir = Path(vdir)
    table = {}
    for mdir in sorted(vdir.glob("mode*")):
        mode = int(mdir.name[4:])
        chunks = []
        for c in sorted(mdir.glob("chunk*")):
            m = re.match(r"chunk\d+_(\d+)-(\d+)", c.name)
            if not m or not (c / "abcrown.log").exists():
                continue
            st, en = int(m.group(1)), int(m.group(2))
            res, t = chunk_status(c / "abcrown.log")
            ok = len(res) > 0 and all(r == "safe" for r in res) and len(res) == len(list((c / "specs").glob("*.vnnlib")))
            chunks.append((en - st, st, en, ok, t))
        n_var = max((en for _, _, en, _, _ in chunks), default=0)
        status = {}
        for size, st, en, ok, _ in sorted(chunks, reverse=True):  # coarse first, finer evidence overrides
            for v in range(st, en):
                status[v] = "certified" if ok else status.get(v, "unresolved")
                if ok:
                    status[v] = "certified"
        for size, st, en, ok, _ in sorted(chunks):  # finest chunks have the final word
            for v in range(st, en):
                if ok:
                    status[v] = "certified"
        cert = sum(1 for v in range(n_var) if status.get(v) == "certified")
        table[mode] = dict(variants=n_var, nominal_certified=status.get(0) == "certified",
                           faults_certified=cert - (1 if status.get(0) == "certified" else 0),
                           faults_total=n_var - 1,
                           unresolved=[v for v in range(n_var) if status.get(v) != "certified"],
                           verifier_seconds=round(sum(t for *_, t in chunks), 1))
    print(json.dumps(table, indent=1))
    (vdir / "fault_table.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
