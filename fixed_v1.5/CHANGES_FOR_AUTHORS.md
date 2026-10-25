# CLadder v1.5: issues we found and how we fixed them

*Samuel Simko, with Raimondas Lencevicius. October 2026.*

We have been using CLadder a lot, and while checking it carefully we found some problems in the data. This note
lists every change we made, why we made it, and one example of each. Row ids are unchanged everywhere, so every
change can be traced back to the original file. A full list of changes, one line per changed field with the old and
new value, is in `CHANGELOG.csv`.

**The short version.** Most of the dataset is fine. The main problems are:

1. a small number of answer keys are wrong (75 in `full_v1.5_default`);
2. many worked solutions (the `reasoning` field) do not add up, even when the final answer is right;
3. some questions cannot be answered from the numbers in the prompt;
4. the six test files are missing the question sentence.

We fixed what can be fixed. Rows that cannot be fixed are flagged or moved to a separate file, not deleted.

---

## 1. Wrong answer keys

**What.** 75 labels in `full_v1.5_default` are wrong. They come from two separate bugs.

**(a) Approximate inference.** For some graphs (arrowhead, IV), the numbers in the prompt and the answer key come
from approximate inference (loopy belief propagation). On these graphs the approximation is not exact, and in 38 rows
it changes the answer. We recomputed everything exactly, by enumerating the full joint distribution of each model.
Raimondas found these 38 independently; our ids and values match his.

*Example (id 1377, marginal).* The prompt says P(medication) = 88%, P(healthy heart | no medication) = 59%,
P(healthy heart | medication) = 49%. Then P(healthy heart) = 0.88 × 0.49 + 0.12 × 0.59 = 0.502, so "Is a healthy
heart more likely than heart problems overall?" is **yes**. The old key said **no**. This row is a near-tie: the
exact value is 0.5001, so it is also flagged `rounding_ambiguous` (see section 4).

This also changed the printed numbers in 270 prompts. Exact inference gives slightly different percentages (for
example, 59% becomes 58% in id 1587).

**(b) A bug in the det-counterfactual generator for the diamondcut graph.** The code computes V3 from X, but the
story says V3 depends on V1. Intervening on X therefore wrongly changes V3. This flips 37 of 158 diamondcut
det-counterfactual answers.

*Example (id 9570).* "The CEO's decision to fire causes the manager not to sign and the director to sign. The
manager signing or the director signing causes the employee to be fired. We observed the CEO decides to retain the
employee. Would the employee be fired if the manager did not sign?" If the CEO retains, the manager signs and the
director does not. If we then force the manager not to sign, nobody signs, so the employee is **not** fired. The old
key said **yes**.

**What we did.** Relabelled all 75. Each row has `label_flipped = True` and keeps the original answer in
`old_label`. Our exact recomputation now agrees with the label in all 6,636 numeric rows of the fixed default file.

## 2. Worked solutions that do not add up

**What.** In many rows the final answer is right but the arithmetic in the `reasoning` field is wrong. In the
original default file, 4,949 of 6,961 arithmetic lines do not evaluate to the result they print (within 0.005). Even
allowing every input to be off by its rounding (±0.005), 2,812 of them cannot give their printed result, and in
1,349 the expression and the printed result have opposite signs. The labels are not
computed from this text, so this does not change answers. But anyone training on, or grading against, the
reasoning text learns arithmetic that does not hold.

The patterns, by query type:

| query type | what is wrong in the original reasoning |
|---|---|
| NDE (mediation, arrowhead, ...) | the weights use P(V2 \| X=1) instead of P(V2 \| X=0), and the differences are taken across the mediator instead of across X |
| marginal | prints "−" where the formula says "+" (all 1,580 rows) |
| ETT on confounding | the formula says P(V1 \| X=1), but the numbers plugged in are P(V1) |
| IV ATE | an operator is missing, e.g. `0.83 * (0.57 - 0.23) 0.17 * (0.79 - 0.43)` |
| backadj | `reasoning` is empty in all 1,580 rows |

*Example (id 4, NDE).* The formula is Σ_v P(V2=v | X=0) · [P(Y | X=1, v) − P(Y | X=0, v)] with P(V2=1 | X=0) = 0.74.
- Original: `0.74 * (0.86 - 0.41) + 0.24 * (0.54 - 0.08) = 0.32`. As written this is 0.44, and it pairs the wrong
  probabilities.
- Fixed: `0.26 * (0.41 - 0.08) + 0.74 * (0.86 - 0.54) = 0.32`.

**What we did.** Rewrote the formula, plug-in and arithmetic lines of every numeric row from the numbers actually
printed in its prompt (5,685 rows in the default file). We also wrote a reasoning for every backadj row: which set
each method adjusts for, which set is a valid backdoor set, and so which method is right (1,580 rows). Every
`expression = result` line now evaluates to its printed result: 6,469 of 6,469 lines in the default file.

## 3. Questions that cannot be answered from the prompt

Some questions need a number that the prompt never gives.

**(a) NDE / NIE on the arrowhead graph (316 rows): moved to a separate file.** These need P(Y | X, V2, V3), which is
not printed. The worked solution teaches a formula based on P(Y | X, V3), which is confounded through V2. That
formula is neither the natural direct effect nor a controlled direct effect, so these questions are ill-posed as
written. Exact recomputation alone does not fix them; they need a new definition first. They are in
`*_excluded_ill_posed.csv` with `ill_posed = True`, not deleted.

**(b) ETT on the confounding graph (158 rows): flagged, plus a fixed variant.** The answer needs P(X | V1), which is
not printed. In `full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv` we add one sentence in the prompts'
own style ("For patients who are not male, the probability of receiving treatment is 93%. ..."). With it, all 158
become answerable and none is rounding-ambiguous.

**(c) IV ATE (158 rows): flagged.** The label is the generator's true ATE, but the prompt only supports the Wald
ratio. With a hidden confounder these are not the same thing. They usually agree (median difference 0.006), but 7 of
158 answers would change under the Wald ratio. These rows have `iv_wald_gives_late_not_ate = True`, and their labels
are kept.

`identifiable_from_prompt` marks all of these mechanically. A row passes only if the identification formula,
evaluated on the exact values of the printed quantities, equals the true answer.

*Note on exp_away (158 rows).* The effect size is not identified as printed, but the yes/no answer is: its sign
equals the sign of P(Y | X=1, V3=1) − P(Y | X=0, V3=1), which is printed. The original reasoning weighted by P(X)
instead of P(X | V3=1). We rewrote it to use this difference.

## 4. Answers decided by rounding

All percentages are printed as whole numbers. In some rows the answer changes depending on how the hidden exact
values were rounded. For example, id 7892 prints P(X) = 98%, and the effect could be anywhere from −0.36 to +0.46.

**What we did.** Flagged them: `rounding_ambiguous = True` if the answer can change anywhere within ±0.5 percentage
points of every printed number (481 rows in the default file, 259 of them correlation questions). We also provide
`full_v1.5_default_variant_1decimal.csv`, with every percentage printed to one decimal. That leaves only 64
ambiguous rows.

## 5. The test files are missing the question

**What.** In all six `test-*-v1.5.csv` files the prompt ends before the question. No row contains a "?". The
question is only in the matching `cladder-v1-q-*.json` files of the v1 release.

**What we did.** Appended each row's question from its JSON, matched by id; ids and order are identical. Every test
prompt now ends with its question. We then applied fixes 1–4 to each test file using its own JSON. Label changes per
file: balanced 75, easy 103, hard 112, noncommonsense 133, anticommonsense 31, commonsense 31.

*Example (test-hard, id 1).* Before: "... For husbands that set the alarm, the probability of ringing alarm is 22%."
After: the same, followed by "Will alarm set by husband decrease the chance of ringing alarm?"

## 6. Duplicates

1,195 rows in the default file repeat another row's prompt word for word. We did not remove them. `duplicate_of`
gives the id of the first copy, and `n_identical_prompts` gives the number of copies.

---

## Files

| file | rows | what it is |
|---|---|---|
| `full_v1.5_default_fixed.csv` | 9,796 | the default file with all fixes, minus the 316 ill-posed rows |
| `full_v1.5_default_excluded_ill_posed.csv` | 316 | the arrowhead NDE/NIE rows (section 3a) |
| `full_v1.5_default_variant_1decimal.csv` | 6,952 | numeric rows with one-decimal percentages (section 4) |
| `full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv` | 158 | confounding ETT with the missing number added (section 3b) |
| `test-*-v1.5_fixed.csv`, `test-*-v1.5_excluded_ill_posed.csv` | | the six test files, same fixes |
| `CHANGELOG.csv` | | every changed field, with old and new value |

New columns: `label_flipped`, `old_label`, `exact_value`, `numbers_rerendered`, `reasoning_regenerated`,
`ill_posed`, `identifiable_from_prompt`, `rounding_ambiguous`, `rounding_ambiguous_1dp`,
`iv_wald_gives_late_not_ate`, `reasoning_end_matches_label`, `duplicate_of`, `n_identical_prompts`.

## How we checked

- **Exact recomputation.** Our own code enumerates each meta-model exactly. It reproduces every stored label from
  the stored ground truth with CLadder's own 0.005 rule before changing anything, and after the fixes it agrees with
  all 6,636 numeric labels.
- **Independent checker.** Raimondas's independent checker finds his 38 wrong keys in the original file and 0 in
  the fixed file. It finds 0 printed numbers that differ from exact inference (270 in the original).
- **Arithmetic.** Every `expression = result` line in the fixed reasoning evaluates to its printed result.
- **LLM check.** Qwen3.8-27B solved the flipped rows and checked a sample of rewritten solutions (results below).

*LLM check results will be added here in a follow-up commit (run in progress: `src/qwen_check.py`).*

## Not changed

- `full_v1.csv`.
- collider_bias, and det-counterfactual rows on graphs other than diamondcut (we did not audit them in depth).
- The generator code itself. Our fixes are applied to the data files; the corresponding generator bugs are pointed
  to above, e.g. `queries/associative.py:53` (marginal sign), `mediators.py:158` (NDE weights) and `ett.py:42`
  (ETT plug-in).
