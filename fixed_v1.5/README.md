# CLadder v1.5, locally fixed (2026-10-20)

Built from CLadder @3d2d1169, the HF v1.5 CSVs, and the v1 release zip (meta models and `cladder-v1-q-*.json`). See `CHANGES_FOR_AUTHORS.md` for a plain-language account of every change. All files are in `data/`.

Rebuild: put the original inputs in `inputs/` (`inputs/hf/data/*.csv` from the HF dataset, `inputs/v1zip/cladder-v1-*.json` from the v1 zip) or set `CLADDER_DL`, then run `python3 -I src/build.py` (stdlib only, ~30 s; writes to `data/`). Check: `python3 -I src/check.py` (needs numpy). Section (b) of the check uses Raimondas Lencevicius's independent checker, which is not distributed here; set `CLADDER_RL_CHECKER` to its `checker/` directory to run it. Its stored results are in `checks/check_report.json`.

## Files
| file | content |
|---|---|
| `full_v1.5_default_fixed.csv` | 9796 rows: the default file minus the ill-posed rows. It keeps the original columns and adds flag columns |
| `full_v1.5_default_excluded_ill_posed.csv` | 316 arrowhead NDE/NIE rows, kept out of the default file rather than dropped silently |
| `full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv` | the 158 confounding-ETT rows with P(X\|V1) printed (fix 3) |
| `full_v1.5_default_variant_1decimal.csv` | the 6952 numeric rows (including the 316 ill-posed arrowhead NDE/NIE rows, flagged `ill_posed`) with every percentage printed to one decimal and derivations to 3 decimals (fix 4) |
| `test-*-v1.5_fixed.csv`, `test-*-v1.5_excluded_ill_posed.csv` | all 6 test files, run through the same pipeline using each file's own JSON |
| `full_v1.5_default_clean.csv`, `test-*-v1.5_clean.csv` | the **clean benchmark**: the fixed rows minus every ill-posed one (not identifiable from the prompt, rounding-ambiguous, or arrowhead NDE/NIE). 9,021 rows in default; `checks/clean_manifest.json` has per-file counts. Build: `python src/make_clean.py` |
| `CHANGELOG.csv` | one line per (file, id, fix, field) with the old and new value |
| `src/exact_ours.py` | our enumeration engine (from `verify/exact.py`) |

New columns:
- `ill_posed`
- `identifiable_from_prompt`
- `rounding_ambiguous`: box over the printed integer %
- `rounding_ambiguous_1dp`
- `iv_wald_gives_late_not_ate`
- `numbers_rerendered`
- `reasoning_regenerated`
- `label_flipped`
- `old_label`
- `exact_value`: the exact CLadder-recipe estimand
- `reasoning_end_matches_label`
- `duplicate_of`
- `n_identical_prompts`

## What was fixed (default file; ids kept)
1. **Exact inference.**
   - Every printed quantity in `meta.given_info` is recomputed by full joint enumeration of the meta-model world and re-rendered with `.0%`. This changes **270 prompts**: IV ate 72, arrowhead ate 73, arrowhead ett 60, arrowhead marginal 65.
   - Labels are recomputed from the exact CLadder-recipe estimand with CLadder's own rule (0.005 cut). The rule was first checked to reproduce all stored labels from the stored groundtruth (it is asserted per row).
   - **38 labels flip:** marginal 23, nie 10, nde 4, ett 1. These are the same 38 as Raimondas's kind 1. 14 of the 38 (the arrowhead NDE/NIE) end up in the excluded file.
2. **Arrowhead NDE/NIE:** flagged `ill_posed` and moved to the excluded file. Their reasoning says "not identified".
3. **Kind 2.**
   - `identifiable_from_prompt` is set mechanically. A row passes only if the identification formula, evaluated at the exact values of the printed quantities, equals the exact estimand to 1e-9.
   - It is False for confounding ETT (158) and IV ATE (158: Wald ≠ the generator's ATE, see fix 7). It is also False for all excluded arrowhead NDE/NIE.
   - exp_away (158) is True. Its answer depends only on the sign, and by convexity sign(P(Y|X=1,V3=1) − P(Y|V3=1)) = sign(P(Y|X=1,V3=1) − P(Y|X=0,V3=1)), which is printed (`cladder_compile/proofs/PROOFS.md`). The reasoning now uses this difference instead of CLadder's taught formula, which wrongly weights by P(X) instead of P(X|V3=1). The check for this cell is sign agreement with the exact value, not value equality.
   - In the variant file I added this sentence, in the prompts' own template style: "For {V1=0 phrase}, the probability of {X1} is k%. For {V1=1 phrase}, the probability of {X1} is k%."
   - With that sentence, all 158 become identified (formula = exact ETT) and 0 of 158 are rounding-ambiguous.
4. **Kind 3:** `rounding_ambiguous` means the identification formula is evaluated over all corners of the ±0.5-point box on the printed integer %. An IV denominator interval that contains 0 counts as unbounded.
   - **481 flagged:** correlation 259, ate 79, marginal 56, ett 34, nie 29, exp_away 20, nde 4. Of these, 459 are in rows that are otherwise identifiable.
   - Printing one decimal leaves **64**: correlation 31, ate 16, ett 9, marginal 5, nie 3.
   - This is a plain box with no joint-consistency or builder-support constraint, so it is a superset of Raimondas's certified 340.
5. **Derivations.**
   - For every numeric row, step3 (formula), step4 (the printed numbers as 2-dp) and step5 (arithmetic) are regenerated from the prompt's printed numbers, and `end` is regenerated with them.
   - This fixes the marginal sign, the NDE weights and differencing, the IV missing operator, and the ETT-confounding P(V1\|X=1). Where that quantity isn't printed, the step says it is not identified.
   - Backadj reasoning is now filled in all 1580 rows: which set each method adjusts for, which set is valid, and which method is right.
   - In total 5685 numeric reasonings and 1580 backadj reasonings changed.
6. **det-counterfactual diamondcut:** V3 is now computed from V1 ("V3 = V1" / "V3 = not V1"), as the prompt states, and step4, step5 and end are rewritten in all 158 rows. **37 labels flip, the same id set as `v15_ref_label_disagree.csv`** (symmetric difference empty).
7. **IV ATE:** the `iv_wald_gives_late_not_ate` flag is set on all 158 rows. Labels follow the generator's exact ATE.
   - Wald and that ATE differ: the median |diff| is 0.006, and 7 of 158 answers would differ under exact Wald.
   - Some rows have a near-zero denominator, where the difference reaches 24.6.
8. **Test files:** the question sentence from the matching `cladder-v1-q-<name>.json` (matched by id; ids and order equal) is appended. Each file then gets the same fixes 1-7 and 9. Label flips per file: balanced 75, easy 103, hard 112, noncommonsense 133, anticommonsense 31, commonsense 31.
9. **Duplicates:** `duplicate_of` (id of the first identical prompt) and `n_identical_prompts`. There are 1195 duplicate rows in default.

## Check results
- **(a)** Our exact recomputation agrees with the label in **6636 of 6636** numeric rows of the fixed default file (0 mismatches).
- **(b)** Raimondas's `checker/exact.py` (`estimand` with `'exact'`, `answer`, `polarity` parsed from the prompt, `render`/`printed_numbers`):

  | file | kind 1 | printed numbers ≠ exact `.0%` |
  |---|---|---|
  | original default (sanity) | **38** (his 38, same cells) | 270 |
  | fixed default | **0** | **0** |
  | excluded file | 0 | 0 |
  | ETT variant | 0 | 158 (expected: his `quantities()` doesn't know the added sentence) |

  - Kind 2 on the fixed file is 316 rows (confounding ETT and IV ATE, 158 each) not identified from the prompt, plus the 316 excluded. Kind 3 is 481 by box, 64 with one decimal.
- **(c)** Every `expr = result` line in the fixed reasoning evaluates to its printed result within 0.005:
  - fixed file: 6469 of 6469 lines, 0 unparsable;
  - ETT variant: 158 of 158;
  - 1-decimal variant: 6477 of 6477 within 0.0005;
  - the original file had 4949 of 6961 off and 158 unparsable.
  - 9 rows have no arithmetic line because a printed denominator is 0% (correlation 8, IV ate 1), and the 158 confounding-ETT rows say "not identified".
- **(d)** All 6 test files: 0 prompts without "?", and every prompt ends with its own JSON question. test-balanced matches the default prompt and label for 10112 of 10112 ids.
  - In the other 5 files, the ids refer to different questions than the same id in default (only ~50% label agreement on shared ids), so id-matching to default is not meaningful there.
  - 3189-4747 prompts per file occur verbatim in default.
- **(e)** Rows changed per fix (default): F1 numbers 270, F1 labels 38, F5 reasoning 5685 (+1580 backadj), F6 detcf reasoning 158 and labels 37. Label flips by query_type: det-counterfactual 37, marginal 23, nie 10, nde 4, ett 1, total 75. The ids are in `checks/check_report.json`.

## Known remaining issues
- `reasoning_end_matches_label=False` in 125 rows: the plug-in value from the printed numbers lands on the other side of the line from the exact label. All of them are flagged `rounding_ambiguous` or not identifiable. Use the 1-decimal variant to shrink this.
- exp_away: the effect size is not identified as printed (P(Y|V3=1) needs P(X|V3=1)), but the yes/no answer is, by convexity. Fixed in v2, see fix 3.
- The IV ATE is only bounded with a hidden confounder. Labels are kept on the generator's ATE, and the rows are only flagged.
- Arrowhead NDE/NIE need redefinition before reuse. No target was chosen.
- The kind-3 box is conservative (no joint-consistency constraints), and the cut-only cases are included.
- Not touched:
  - the odd step4 labels in the correlation rows of the original (`P(X=1=1)`); they are now regenerated for numeric rows;
  - collider_bias and the det-counterfactual rows on graphs other than diamondcut;
  - `full_v1.csv`;
  - the 58 CEO-story ATE rows where one phrase names two values.
- Derivation fixes are template rewrites in this pipeline, not patches to CLadder's generator code.

## v2 (2026-10-20, coordinator)
exp_away corrected: sign identified by convexity, reasoning uses P(Y|X=1,V3=1)-P(Y|X=0,V3=1). Rebuilt with src/build.py and re-checked with src/check.py. Pre-fix outputs: checks/*.before_expaway.json, src/build.py.bak_expaway.

## v3 (2026-10-25)
- backadj reasoning no longer starts with the literal string `nan nan` (the source stores empty steps as 'nan'; build.py now drops them). Check report otherwise unchanged.
- README: the 1-decimal variant holds the numeric rows including the ill-posed ones (it said 'non-excluded').
- Independent LLM check with Qwen3.8-27B: src/qwen_check.py, results in checks/qwen_check_*.csv.
