"""Build the REPAIRED CLadder v1.5 benchmark: fix ill-posed rows instead of dropping them, where a prompt edit can.

Two of the three ill-posed classes are fixed by re-rendering the prompt, not by removing the row:
  rounding-ambiguous  -> print every percentage to one decimal (full_v1.5_default_variant_1decimal.csv). This makes
                         the answer stable for 417 of the 481 rows; 64 still flip at one decimal and are dropped.
  confounding-ETT     -> add the one missing line, P(X|V1), to the prompt
                         (full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv). All 158 become identifiable.
The third class cannot be fixed by editing the prompt and is left out:
  IV-ATE (158)        -> with the hidden confounder the Wald ratio the prompt identifies is the LATE, not the ATE the
                         label uses; adding a number would break the instrument. These stay out (as do the arrowhead
                         NDE/NIE, which need the estimand redefined, section 3a).

Result: full_v1.5_default_repaired.csv. Every row is identifiable from its (possibly re-rendered) prompt and stable
under rounding. Run from the repo root after the fixed/variant files exist:
  python -m src.make_repaired
"""
import os

import pandas as pd

HERE = os.environ.get("CLADDER_OUT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf = lambda s: s.astype(str).str.lower().isin(["true", "1"])
R = lambda n: pd.read_csv(os.path.join(HERE, n), low_memory=False)


def main():
    fx = R("full_v1.5_default_fixed.csv")
    d1 = R("full_v1.5_default_variant_1decimal.csv").set_index("id")
    et = R("full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv")
    cols = list(fx.columns)

    well = lambda d: (tf(d.identifiable_from_prompt) & ~tf(d.rounding_ambiguous) & ~tf(d.ill_posed)
                      & ~tf(d.iv_wald_gives_late_not_ate))
    out = {r.id: r for _, r in fx[well(fx)].iterrows()}          # the already-clean rows
    base = len(out)

    # fix 1: rounding. a row dropped only for integer-rounding ambiguity, now stable at one decimal, re-enters with
    # its one-decimal prompt and reasoning.
    nr = 0
    for rid in fx.loc[tf(fx.rounding_ambiguous) & tf(fx.identifiable_from_prompt) & ~tf(fx.ill_posed)
                      & ~tf(fx.iv_wald_gives_late_not_ate), "id"]:
        if rid in out or rid not in d1.index:
            continue
        row = d1.loc[rid]
        if tf(pd.Series([row.rounding_ambiguous_1dp])).iloc[0]:   # still flips at one decimal -> leave out
            continue
        r = row.copy(); r["id"] = rid; out[rid] = r; nr += 1

    # fix 2: confounding-ETT. swap in the +P(X|V1) prompt; all 158 are now identifiable and not rounding-ambiguous.
    ne = 0
    for _, row in et.iterrows():
        out[row.id] = row; ne += 1

    rep = pd.DataFrame([out[i] for i in out]).reindex(columns=cols)
    # a repaired benchmark must be fully well-posed
    assert int((~tf(rep.identifiable_from_prompt) | tf(rep.rounding_ambiguous_1dp.fillna(rep.rounding_ambiguous))
                | tf(rep.ill_posed) | tf(rep.iv_wald_gives_late_not_ate)).sum()) == 0, "repaired set still has bad rows"
    rep.to_csv(os.path.join(HERE, "full_v1.5_default_repaired.csv"), index=False)
    print(f"repaired default: {len(rep)} rows (clean base {base} + rounding-repaired {nr} + ETT-repaired {ne})")
    print(f"  labels: {dict(rep.label.value_counts())}")
    print(f"  still dropped: IV-ATE {int(tf(fx.iv_wald_gives_late_not_ate).sum())}, "
          f"rounding-at-1-decimal {int(tf(fx.rounding_ambiguous).sum()) - nr - 0}, "
          f"arrowhead NDE/NIE (excluded file, need redefinition)")


if __name__ == "__main__":
    main()
