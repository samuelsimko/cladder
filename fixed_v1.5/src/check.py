"""Checks on the fixed files. Run from fixed_v1.5/: python3 -I src/check.py

Section (b) uses Raimondas Lencevicius's independent checker (checker/exact.py, not distributed here). Point
CLADDER_RL_CHECKER at a copy of his checker/ directory to run it (needs numpy); otherwise (b) is skipped and the stored
results in checks/check_report.json remain the reference."""
import csv, json, os, re, sys, collections
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RL = None
if os.environ.get('CLADDER_RL_CHECKER'):
    sys.path.insert(0, os.environ['CLADDER_RL_CHECKER'])
    import exact as RL  # Raimondas Lencevicius's checker/exact.py, unmodified
import build as B   # our engine
csv.field_size_limit(10**9)
OUT = B.OUT; DL = B.DL
META = B.META
rep = {}

def load(p): return list(csv.DictReader(open(p)))
fixed = load(f'{OUT}/full_v1.5_default_fixed.csv')
excl = load(f'{OUT}/full_v1.5_default_excluded_ill_posed.csv')
orig = load(f'{DL}/hf/data/full_v1.5_default.csv')
qj = {x['question_id']: x for x in json.load(open(f'{DL}/v1zip/cladder-v1-q-balanced.json'))}

# (a) our exact recomputation vs every label in the fixed default file
bad_a = collections.Counter(); n_a = 0
for r in fixed:
    mt = qj[int(r['id'])]['meta']; qt = mt['query_type']
    if qt in B.NUMERIC:
        n_a += 1
        if B.answer(qt, B.exact_estimand(qt, mt['model_id'], mt), mt) != r['label']: bad_a[qt] += 1
rep['a_our_exact_vs_label'] = dict(numeric_rows=n_a, mismatches=dict(bad_a))

# (b) RL's checker: kind 1 = exact recipe answer != label; plus printed numbers == exact '.0%'
def rl_kind1(rows):
    k1 = collections.Counter(); pn = collections.Counter(); npol = 0; n = 0; ids = []
    for r in rows:
        mt = qj[int(r['id'])]['meta']; qt = mt['query_type']
        if qt not in RL.NUMERIC_TYPES: continue
        m = RL.Model.from_params(META[mt['model_id']]['params']); g = RL.graph_of(m)
        assert g == mt['graph_id']
        pol = RL.polarity(qt, r['prompt'])
        if pol is None: npol += 1; continue
        n += 1
        v = RL.estimand(qt, m, 'exact')
        if RL.answer(qt, v, pol) != r['label']: k1[(qt, g)] += 1; ids.append(r['id'])
        qs = RL.quantities(qt, g)
        if qs is not None and [RL.pct_int(x) for x in RL.render(m, qs, 'exact')] != RL.printed_numbers(r['prompt']):
            pn[(qt, g)] += 1
    return dict(rows_checked=n, polarity_unparsed=npol, kind1=sum(k1.values()), kind1_by_cell={str(k): v for k, v in k1.items()},
                kind1_ids=ids[:60], printed_numbers_not_exact=sum(pn.values()), printed_by_cell={str(k): v for k, v in pn.items()})
var = load(f'{OUT}/full_v1.5_default_variant_confounding_ett_with_PXgivenV1.csv')
if RL is not None:
    rep['b_RL_checker_original_default'] = rl_kind1(orig)
    rep['b_RL_checker_fixed_default'] = rl_kind1(fixed)
    rep['b_RL_checker_excluded_file'] = rl_kind1(excl)
    rep['b_RL_checker_variant_ett'] = rl_kind1(var)
# kind 2 / kind 3 counts (our flags; kind 3 = box over printed integer %)
def fl(rows, k): return collections.Counter(r['query_type'] for r in rows if r[k] == 'True')
rep['kind2_not_identifiable_fixed'] = dict(fl(fixed, 'identifiable_from_prompt') and
    collections.Counter((r['query_type'], r['graph_id']) for r in fixed if r['identifiable_from_prompt'] == 'False'))
rep['kind2_not_identifiable_fixed'] = {str(k): v for k, v in rep['kind2_not_identifiable_fixed'].items()}
rep['kind2_variant_ett_identified'] = collections.Counter(r['identifiable_from_prompt'] for r in var)
rep['kind2_variant_ett_rounding_ambiguous'] = collections.Counter(r['rounding_ambiguous'] for r in var)
rep['kind3_rounding_ambiguous_fixed'] = dict(fl(fixed, 'rounding_ambiguous')); rep['kind3_total'] = sum(rep['kind3_rounding_ambiguous_fixed'].values())
v1 = load(f'{OUT}/full_v1.5_default_variant_1decimal.csv')
v1 = [r for r in v1 if r['ill_posed'] == 'False']
rep['kind3_ambiguous_1decimal'] = dict(fl(v1, 'rounding_ambiguous_1dp')); rep['kind3_1dp_total'] = sum(rep['kind3_ambiguous_1decimal'].values())
rep['kind3_ambiguous_among_identifiable'] = sum(1 for r in fixed if r['rounding_ambiguous'] == 'True' and r['identifiable_from_prompt'] == 'True')
rep['iv_wald_flag'] = sum(r['iv_wald_gives_late_not_ate'] == 'True' for r in fixed)
rep['reasoning_end_disagrees_label'] = dict(collections.Counter(r['query_type'] for r in fixed if r['reasoning_end_matches_label'] == 'False'))

# (c) every arithmetic line evaluates to its printed result within 2-dp rounding
pat = re.compile(r'^([0-9\.\s\*\+\-\/\(\)]+)=\s*(-?[0-9]*\.?[0-9]+)\s*$')
def arith(rows, tol):
    tot = collections.Counter(); bad = collections.Counter(); badids = []; unparsed = 0
    for r in rows:
        for l in r['reasoning'].split('\n'):
            m = pat.match(l.strip())
            if not m: continue
            lhs = m.group(1)
            if not re.search(r'[\*\+\/]|\d\s*-\s*\d', lhs): continue
            tot[r['query_type']] += 1
            try: v = eval(lhs, {'__builtins__': {}})
            except SyntaxError: unparsed += 1; bad[r['query_type']] += 1; badids.append(r['id']); continue
            except ZeroDivisionError: tot['zerodiv'] += 1; continue
            if abs(v - float(m.group(2))) > tol + 1e-9: bad[r['query_type']] += 1; badids.append(r['id'])
    return dict(lines=dict(tot), total=sum(tot.values()), bad=dict(bad), unparsed=unparsed, bad_ids=badids[:20])
rep['c_arith_original'] = arith(orig, 0.005)
rep['c_arith_fixed'] = arith(fixed, 0.005)
rep['c_arith_variant_ett'] = arith(var, 0.005)
rep['c_arith_variant_1dp'] = arith(v1, 0.0005)
rep['c_empty_reasoning_fixed'] = sum(1 for r in fixed if not r['reasoning'].strip())
# numeric rows without any arithmetic line
rep['c_numeric_without_arith_line'] = dict(collections.Counter(r['query_type'] for r in fixed if r['query_type'] in B.NUMERIC and not any(pat.match(l.strip()) for l in r['reasoning'].split('\n'))))

# (d) test files
dflt = {r['id']: r for r in fixed + excl}
td = {}
for s in ('anticommonsense', 'balanced', 'commonsense', 'easy', 'hard', 'noncommonsense'):
    rows = load(f'{OUT}/test-{s}-v1.5_fixed.csv') + load(f'{OUT}/test-{s}-v1.5_excluded_ill_posed.csv')
    noq = sum('?' not in r['prompt'] for r in rows)
    ind = [r for r in rows if r['id'] in dflt]
    same_prompt = sum(r['prompt'] == dflt[r['id']]['prompt'] for r in ind)
    same_label = sum(r['label'] == dflt[r['id']]['label'] for r in ind)
    td[s] = dict(rows=len(rows), without_qmark=noq, ids_in_default=len(ind), prompt_equal=same_prompt, label_equal=same_label)
rep['d_testfiles'] = td

# (e) diff summary
cl = load(f'{OUT}/CHANGELOG.csv')
rep['e_rows_changed_per_fix_default'] = dict(collections.Counter(x['fix'] for x in cl if x['file'] == 'full_v1.5_default'))
qt_of = {r['id']: (r['query_type'], r['graph_id']) for r in orig}
flips = [x for x in cl if x['file'] == 'full_v1.5_default' and x['fix'] in ('F1_exact_label', 'F6_detcf_label')]
rep['e_label_flips_by_query_type'] = dict(collections.Counter(qt_of[x['id']][0] for x in flips))
rep['e_label_flips_by_qt_graph'] = {str(k): v for k, v in collections.Counter(qt_of[x['id']] for x in flips).items()}
rep['e_label_flip_ids'] = [x['id'] for x in flips]
rep['e_flips_in_excluded'] = sum(1 for x in flips if x['id'] in {r['id'] for r in excl})
rep['e_rerendered_by_qt'] = dict(collections.Counter(qt_of[x['id']][0] for x in cl if x['file'] == 'full_v1.5_default' and x['fix'] == 'F1_exact_numbers'))
rep['e_changes_per_file'] = dict(collections.Counter((x['file'], x['fix']) for x in cl).most_common()) and {f'{a}|{b}': c for (a, b), c in collections.Counter((x['file'], x['fix']) for x in cl).items()}
rep['dups_fixed_default'] = sum(1 for r in fixed + excl if r['duplicate_of'])
json.dump(rep, open(os.path.join(HERE, 'checks', 'check_report.json' if RL is not None else 'check_report_without_rl.json'), 'w'), indent=1, default=str)
print(json.dumps(rep, indent=1, default=str))
