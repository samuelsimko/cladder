"""Build the CLEAN CLadder v1.5 benchmark: only well-posed rows.

A row is well-posed when its yes/no answer can be derived from the numbers printed in the prompt and does not
depend on how those numbers were rounded. From each *_fixed.csv we drop every row that is any of:
  ill_posed                      (arrowhead NDE/NIE; already in the excluded file, belt and braces here)
  not identifiable_from_prompt   (confounding ETT needs P(X|V1); IV ATE's Wald ratio != the labelled ATE)
  rounding_ambiguous             (the answer flips within +-0.5 percentage points of the printed integers)
Backadj rows have no numbers, are never flagged, and are kept. The result is *_clean.csv next to the fixed files.

Run from the repo root (reads the already-built *_fixed.csv in this directory):
  python -m src.make_clean            # or: python src/make_clean.py
"""
import glob, json, os

import pandas as pd

HERE = os.environ.get("CLADDER_OUT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf = lambda s: s.astype(str).str.lower().isin(["true", "1"])


def clean(df):
    bad = (~tf(df.identifiable_from_prompt)) | tf(df.rounding_ambiguous) | tf(df.ill_posed)
    return df[~bad].copy()


def main():
    manifest = {}
    for f in sorted(glob.glob(os.path.join(HERE, "*_fixed.csv"))):
        name = os.path.basename(f).replace("_fixed.csv", "")
        d = pd.read_csv(f, low_memory=False)
        c = clean(d)
        # a clean benchmark must contain no ill-posed row, by any of the three flags
        assert int(((~tf(c.identifiable_from_prompt)) | tf(c.rounding_ambiguous) | tf(c.ill_posed)).sum()) == 0
        out = os.path.join(HERE, f"{name}_clean.csv")
        c.to_csv(out, index=False)
        manifest[name] = {"fixed": len(d), "clean": len(c), "dropped": len(d) - len(c),
                          "dropped_not_identifiable": int((~tf(d.identifiable_from_prompt)).sum()),
                          "dropped_rounding_ambiguous": int(tf(d.rounding_ambiguous).sum()),
                          "dropped_ill_posed": int(tf(d.ill_posed).sum()),
                          "clean_by_query_type": {k: int(v) for k, v in c.query_type.value_counts().items()}}
        print(f"{name:34s} fixed {len(d):5d} -> clean {len(c):5d}  (-{len(d) - len(c)})")
    json.dump(manifest, open(os.path.join(HERE, "checks", "clean_manifest.json"), "w"), indent=1)
    print("wrote *_clean.csv and checks/clean_manifest.json")


if __name__ == "__main__":
    main()
