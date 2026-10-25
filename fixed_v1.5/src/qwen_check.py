"""Independent LLM check of the fixed CLadder files with Qwen3.8-27B (no OpenRouter key available).

Two checks, thinking mode ON (the model works the arithmetic itself), temperature 0:
  A. SOLVE   the model answers the fixed prompt (yes/no). Samples: every label-flipped row of the fixed default file
             and of its excluded file, plus a random control of unchanged numeric and backadj rows. On flipped rows
             we report how often the model agrees with the NEW label vs the OLD one.
  B. VERIFY  the model reads the prompt and the REWRITTEN reasoning and says whether every step is correct and
             follows from the prompt (OK / ERROR). Samples: rows whose reasoning was regenerated (numeric + backadj).
Writes checks/qwen_check_{solve,verify}.csv and prints a summary.

  python src/qwen_check.py --model /dev/shm/ssimko_q27b [--n_control 300 --n_verify 300]
"""
import argparse, os, re

import pandas as pd
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/ssimko_q27b")
    ap.add_argument("--n_control", type=int, default=300); ap.add_argument("--n_verify", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=6000)
    a = ap.parse_args()
    D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    fx = pd.read_csv(f"{D}/full_v1.5_default_fixed.csv", low_memory=False)
    ex = pd.read_csv(f"{D}/full_v1.5_default_excluded_ill_posed.csv", low_memory=False)
    tf = lambda s: s.astype(str).str.lower().isin(["true", "1"])
    flip = pd.concat([fx[tf(fx.label_flipped)].assign(src="fixed"), ex[tf(ex.label_flipped)].assign(src="excluded")])
    ctrl = fx[~tf(fx.label_flipped)].sample(a.n_control, random_state=0).assign(src="control")
    solve = pd.concat([flip.assign(kind="flip"), ctrl.assign(kind="control")], ignore_index=True)
    ver = fx[tf(fx.reasoning_regenerated)].sample(a.n_verify, random_state=1)

    llm = LLM(a.model, max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=128, trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    chat = lambda m: tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    S = ("Answer the following causal reasoning question. Use only the information given. Work it out carefully, then "
         "end with exactly one final line: ANSWER: yes  or  ANSWER: no\n\n")
    V = ("Below is a causal reasoning question and a worked solution. Check the worked solution step by step: are the "
         "formula, the numbers plugged in (they must come from the question), the arithmetic, and the final yes/no all "
         "correct? Small rounding differences (<= 0.01) are fine. End with exactly one final line: VERDICT: OK  or  "
         "VERDICT: ERROR - <one short reason>\n\nQUESTION:\n{p}\n\nWORKED SOLUTION:\n{r}")
    last = lambda t, rx: (re.findall(rx, t.split("</think>")[-1], re.I) or [""])[-1].lower()

    out = llm.generate([chat(S + p) for p in solve.prompt.astype(str)], sp)
    solve["model"] = [last(o.outputs[0].text, r"ANSWER:\s*(yes|no)") for o in out]
    solve[["id", "kind", "src", "query_type", "graph_id", "label", "old_label", "rounding_ambiguous", "exact_value",
           "model"]].to_csv(f"{os.path.dirname(D)}/checks/qwen_check_solve.csv", index=False)
    out = llm.generate([chat(V.format(p=p, r=r)) for p, r in zip(ver.prompt.astype(str), ver.reasoning.astype(str))], sp)
    ver["verdict"] = [last(o.outputs[0].text, r"VERDICT:\s*(OK|ERROR[^\n]*)") for o in out]
    ver[["id", "query_type", "graph_id", "label", "verdict"]].to_csv(f"{os.path.dirname(D)}/checks/qwen_check_verify.csv", index=False)

    print("\n## A. SOLVE")
    for (k, s), g in solve.groupby(["kind", "src"]):
        ans = g.model != ""
        print(f"{k:8s} {s:9s} n={len(g):4d} answered {ans.mean():.2f} | agrees NEW label {(g.model == g.label)[ans].mean():.3f}"
              + (f" | agrees OLD label {(g.model == g.old_label)[ans].mean():.3f}" if k == "flip" else ""))
    f = solve[solve.kind == "flip"]
    for ra, g in f.groupby(tf(f.rounding_ambiguous)):
        ans = g.model != ""
        print(f"  flips rounding_ambiguous={ra}: n={len(g)} NEW {(g.model == g.label)[ans].mean():.3f} OLD {(g.model == g.old_label)[ans].mean():.3f}")
    print(f.groupby("query_type").apply(lambda g: f"n={len(g)} NEW={(g.model == g.label).mean():.2f}", include_groups=False).to_string())
    print("\n## B. VERIFY rewritten reasoning")
    print(ver.verdict.str.startswith("ok").groupby(ver.query_type).agg(["size", "mean"]).to_string())
    print("overall OK", round(ver.verdict.str.startswith("ok").mean(), 3), "| no verdict", round((ver.verdict == "").mean(), 3))


if __name__ == "__main__":
    main()
