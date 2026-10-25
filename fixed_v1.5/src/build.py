"""Build a locally fixed CLadder v1.5.  Usage: python3 -I build.py   (CPU, stdlib only)
Inputs (CLADDER_DL, default fixed_v1.5/inputs) are read-only; outputs go to CLADDER_OUT (default fixed_v1.5/data).
"""
import csv, json, re, sys, os, itertools, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_ours import parse, Net, quantities as exq  # our independent enumeration engine (verify/exact.py)

csv.field_size_limit(10**9)
# DL: the original CLadder inputs -- hf/data/*.csv (HF v1.5) and v1zip/cladder-v1-*.json (v1 release zip)
DL = os.environ.get('CLADDER_DL', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
OUT = os.environ.get('CLADDER_OUT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
META = json.load(open(f'{DL}/v1zip/cladder-v1-meta-models.json'))
DETCF_LIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checks', 'detcf_reference_ids.csv')
CUT = 0.005
NUMERIC = ('marginal', 'ate', 'ett', 'nde', 'nie', 'correlation', 'exp_away')

# ---------------------------------------------------------------- exact values
_cache = {}
def world(mid):
    if mid not in _cache:
        m = META[mid]; assert m['model_id'] == mid
        n = Net(parse(m)); q = exq(m)
        _cache[mid] = (n, q)
    return _cache[mid]

def cond(n, var, given):
    tot = 0.0; acc = 0.0
    for st, pr in n.joint():
        if all(st[k] == v for k, v in given.items()):
            tot += pr
            if st[var]: acc += pr
    return acc / tot

def joint(n, given):
    return sum(pr for st, pr in n.joint() if all(st[k] == v for k, v in given.items()))

KEY = re.compile(r'^[pP]\((\w+)(?:=1)?(?:\s*\|\s*(.*))?\)$')
JKEY = re.compile(r'^P\((\w+)=1, (\w+)=(\d)\)$')

def exact_key(n, key):
    """exact value (nested like the stored one) of a given_info key."""
    if key == 'P(Y = 1 | X = 1, V3 = 1) - P(Y = 1 | X = 0, V3 = 1)':
        return cond(n, 'Y', {'X': 1, 'V3': 1}) - cond(n, 'Y', {'X': 0, 'V3': 1})
    mj = JKEY.match(key)
    if mj:
        return joint(n, {mj.group(1): 1, mj.group(2): int(mj.group(3))})
    mk = KEY.match(key)
    var = mk.group(1); pa = [p.strip() for p in mk.group(2).split(',')] if mk.group(2) else []
    if not pa:
        return cond(n, var, {})
    def rec(fixed, rest):
        if not rest:
            return cond(n, var, dict(fixed))
        return [rec(fixed + [(rest[0], b)], rest[1:]) for b in (0, 1)]
    return rec([], pa)

def flat(v):
    if isinstance(v, list):
        out = []
        for e in v: out += flat(e)
        return out
    return [float(v)]

def unflat(template, vals):
    it = iter(vals)
    def rec(t):
        return [rec(e) for e in t] if isinstance(t, list) else next(it)
    return rec(template)

def exact_estimand(qt, mid, mt):
    n, q = world(mid)
    if qt == 'marginal': return q['obs']['Y']
    if qt == 'ate': return q['ATE']
    if qt == 'ett': return q['ETT_recipe']
    if qt == 'nde': return q['NDE_recipe']
    if qt == 'nie': return q['NIE_recipe']
    if qt == 'correlation': return q['PY1_X1'] - q['PY1_X0']
    if qt == 'exp_away':
        return cond(n, 'Y', {'X': 1, 'V3': 1}) - cond(n, 'Y', {'V3': 1})
    raise KeyError(qt)

# ---------------------------------------------------------------- CLadder answer rule
def answer(qt, gt, mt):
    pol = bool(mt.get('polarity', True)); tr = bool(mt.get('treated', True)); res = bool(mt.get('result', True))
    sg = 0 if gt == 0 else (1 if gt > 0 else -1)
    if qt == 'marginal':  # associative.py:113-120 (cut applied to P itself)
        s = 0 if gt == 0.5 else (1 if gt > 0.5 else -1)
        a = -1 if (s == 0 or abs(gt) < CUT) else s * (-1) ** ((1 - pol) + (1 - tr))
    elif qt == 'correlation':  # associative.py:260-268
        a = -1 if (sg == 0 or abs(gt) < CUT) else sg * (-1) ** ((1 - pol) + (1 - tr) + (1 - res))
    elif qt == 'exp_away':  # associative.py:443-449, no cut
        a = sg * (-1) ** (pol ^ res)
    elif qt == 'ate':
        if sg == 0: return 'unknown'
        a = -1 if abs(gt) < CUT else sg * (-1) ** (1 - pol)
    elif qt == 'ett':
        if sg == 0: return 'unknown'
        a = -1 if abs(gt) < CUT else sg * (-1) ** (1 + 1 - pol)
    else:  # nde nie
        a = -1 if (sg == 0 or abs(gt) < CUT) else sg * (-1) ** (1 - pol)
    return {-1: 'no', 1: 'yes', 0: 'unknown'}[a]

def line(qt):
    return 0.5 if qt == 'marginal' else 0.0

# ---------------------------------------------------------------- identification formulas (in printed quantities)
# each returns (step3 text, expression string) given G (dict key -> nested values) and number formatter f
def F(qt, graph, G, f):
    k = set(G)
    if qt == 'marginal':
        px = G['P(X)']; y0, y1 = G['P(Y | X)']
        return ('P(Y | X=1)*P(X=1) + P(Y | X=0)*P(X=0)', f'{f(px)}*{f(y1)} + {f(1-px)}*{f(y0)}')
    if qt == 'correlation':
        m = G['P(X=1)']; j0 = G['P(Y=1, X=0)']; j1 = G['P(Y=1, X=1)']
        return ('P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)', f'{f(j1)}/{f(m)} - {f(j0)}/{f(1-m)}')
    if qt == 'exp_away':  # sign of P(Y|X=1,V3=1)-P(Y|V3=1) equals sign of this difference (convexity, PROOFS.md)
        t = G['P(Y | X, V3)']
        return ('sign of P(Y=1 | X=1, V3=1) - P(Y=1 | V3=1) = sign of P(Y=1 | X=1, V3=1) - P(Y=1 | X=0, V3=1), since P(Y=1 | V3=1) lies between the two',
                f'{f(t[1][1])} - {f(t[0][1])}')
    if k == {'p(Y | X)'}:
        y0, y1 = G['p(Y | X)']
        return ('P(Y=1|X=1) - P(Y=1|X=0)', f'{f(y1)} - {f(y0)}')
    if qt == 'ate' and k == {'p(V1)', 'p(Y | V1, X)'}:
        pv = G['p(V1)']; t = G['p(Y | V1, X)']
        return ('\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]',
                f'{f(1-pv)} * ({f(t[0][1])} - {f(t[0][0])}) + {f(pv)} * ({f(t[1][1])} - {f(t[1][0])})')
    if qt == 'ett' and k >= {'p(V1)', 'p(Y | V1, X)'}:
        pv = G['p(V1)']; t = G['p(Y | V1, X)']
        if 'p(X | V1)' not in G:
            return ('\\sum_{V1=v} P(V1=v|X=1)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]', None)
        x0, x1 = G['p(X | V1)']
        q = f'{f(pv)}*{f(x1)}/({f(pv)}*{f(x1)} + {f(1-pv)}*{f(x0)})'
        return ('\\sum_{V1=v} P(V1=v|X=1)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)], with P(V1=1|X=1) = P(V1=1)P(X=1|V1=1)/[P(V1=1)P(X=1|V1=1) + P(V1=0)P(X=1|V1=0)]',
                f'(1 - {q}) * ({f(t[0][1])} - {f(t[0][0])}) + {q} * ({f(t[1][1])} - {f(t[1][0])})')
    if qt in ('ate', 'nie') and k == {'p(V3 | X)', 'p(X)', 'p(Y | X, V3)'}:  # frontdoor
        a0, a1 = G['p(V3 | X)']; px = G['p(X)']; y = G['p(Y | X, V3)']
        return ('\\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]',
                f'({f(a1)} - {f(a0)}) * ({f(px)} * ({f(y[1][1])} - {f(y[1][0])}) + {f(1-px)} * ({f(y[0][1])} - {f(y[0][0])}))')
    if qt == 'ate' and k == {'p(X | V2)', 'p(Y | V2)'}:  # IV, Wald ratio
        x0, x1 = G['p(X | V2)']; y0, y1 = G['p(Y | V2)']
        return ('[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]', f'({f(y1)} - {f(y0)}) / ({f(x1)} - {f(x0)})')
    if qt == 'ett' and k == {'p(V3 | X)', 'p(Y | X, V3)'}:
        a0, a1 = G['p(V3 | X)']; y = G['p(Y | X, V3)']
        if graph == 'frontdoor':
            return ('\\sum_{V3=v} P(Y=1|X=1,V3=v)*[P(V3=v|X=1)-P(V3=v|X=0)]', f'({f(y[1][1])} - {f(y[1][0])}) * ({f(a1)} - {f(a0)})')
        if graph == 'diamondcut':
            return ('\\sum_{V3=v} P(V3=v|X=1)*[P(Y=1|X=1,V3=v) - P(Y=1|X=0,V3=v)]',
                    f'{f(1-a1)} * ({f(y[1][0])} - {f(y[0][0])}) + {f(a1)} * ({f(y[1][1])} - {f(y[0][1])})')
    if qt == 'nde' and k == {'p(V2 | X)', 'p(Y | X, V2)'}:
        a0, a1 = G['p(V2 | X)']; y = G['p(Y | X, V2)']
        return ('\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]',
                f'{f(1-a0)} * ({f(y[1][0])} - {f(y[0][0])}) + {f(a0)} * ({f(y[1][1])} - {f(y[0][1])})')
    if qt == 'nie' and k == {'p(V2 | X)', 'p(Y | X, V2)'}:
        a0, a1 = G['p(V2 | X)']; y = G['p(Y | X, V2)']
        return ('\\sum_{V2 = v} P(Y=1|X =0,V2 = v)*[P(V2 = v | X = 1) - P(V2 = v | X = 0)]',
                f'({f(a1)} - {f(a0)}) * ({f(y[0][1])} - {f(y[0][0])})')
    if qt in ('nde', 'nie') and graph == 'arrowhead':
        return ('not identified: needs P(Y|X,V2,V3), and V2 is unobserved', None)
    raise KeyError((qt, graph, sorted(k)))

def ev(s):
    return eval(s, {'__builtins__': {}})

def box_answers(qt, graph, G, keys_order, half, mt):
    """set of CLadder answers reachable when every printed number moves within +-half (absolute)."""
    fl = []; shapes = []
    for kk in keys_order:
        shapes.append((kk, G[kk])); fl += flat(G[kk])
    vals = []
    for d in itertools.product((-half, half), repeat=len(fl)):
        pert = [min(1.0, max(0.0, v + e)) for v, e in zip(fl, d)]
        it = iter(pert); GG = {}
        for kk, t in shapes:
            n_ = len(flat(t)); GG[kk] = unflat(t, [next(it) for _ in range(n_)])
        _, e = F(qt, graph, GG, repr)
        try:
            vals.append(ev(e))
        except ZeroDivisionError:
            return {'yes', 'no'}, (float('-inf'), float('inf'))
    if qt == 'ate' and graph == 'IV':  # Wald ratio: unbounded if the denominator interval contains 0
        x0, x1 = G['p(X | V2)']
        if (x1 + half) - (x0 - half) >= 0 >= (x1 - half) - (x0 + half):
            return {'yes', 'no'}, (float('-inf'), float('inf'))
    lo, hi = min(vals), max(vals)
    pts = [lo, hi]
    L = line(qt)
    for c in (L, L - CUT, L + CUT, L - CUT + 1e-12, L + CUT - 1e-12, L + CUT + 1e-12, L - CUT - 1e-12):
        if lo <= c <= hi: pts.append(c)
    return {answer(qt, p, mt) for p in pts}, (lo, hi)

# ---------------------------------------------------------------- rendering
PCT = re.compile(r'(?<![\w.])(\d{1,3})%')

def rerender_gi(gi_text, old_vals, new_vals, fmt):
    toks = list(PCT.finditer(gi_text))
    if [format(v, '.0%') for v in old_vals] != [t.group(0) for t in toks]:
        return None
    out = []; last = 0
    for t, v in zip(toks, new_vals):
        out.append(gi_text[last:t.start()]); out.append(format(v, fmt)); last = t.end()
    out.append(gi_text[last:])
    return ''.join(out)

def gi_numbers(G, keys, fmtpct):
    """value actually printed (as a probability) for each flattened entry."""
    out = {}
    for kk in keys:
        out[kk] = unflat(G[kk], [float(format(v, fmtpct)[:-1]) / 100 for v in flat(G[kk])])
    return out

def step4_text(G, keys, nd):
    lines = []
    for kk in keys:
        mk = KEY.match(kk); mj = JKEY.match(kk)
        if mj:
            lines.append(f'P({mj.group(1)}=1, {mj.group(2)}={mj.group(3)}) = {G[kk]:.{nd}f}'); continue
        var = mk.group(1); pa = [p.strip() for p in mk.group(2).split(',')] if mk.group(2) else []
        if not pa:
            lines.append(f'P({var}=1) = {G[kk]:.{nd}f}'); continue
        for bits in itertools.product((0, 1), repeat=len(pa)):
            v = G[kk]
            for b in bits: v = v[b]
            lines.append(f'P({var}=1 | ' + ', '.join(f'{p}={b}' for p, b in zip(pa, bits)) + f') = {v:.{nd}f}')
    return '\n'.join(lines)

def end_text(qt, r, nd):
    L = line(qt)
    if qt == 'marginal':
        return f'{r:.{nd}f} > 0.5' if r > 0.5 else (f'{r:.{nd}f} < 0.5' if r < 0.5 else f'{r:.{nd}f} = 0.5')
    if abs(r) < CUT and qt != 'exp_away':
        return f'{r:.{nd}f}, |value| < 0.005 so treated as 0'
    return f'{r:.{nd}f} > 0' if r > 0 else (f'{r:.{nd}f} < 0' if r < 0 else f'{r:.{nd}f} = 0')

# ---------------------------------------------------------------- confounding ETT: missing sentence
def v1_phrases(gi):
    s = re.findall(r'For (.*?), the probability of ', gi)
    b1, b2, b3, b4 = s[:4]
    for m in re.finditer(' and ', b1):
        A0, B0 = b1[:m.start()], b1[m.end():]
        if not (b2.startswith(A0 + ' and ') and b3.endswith(' and ' + B0)): continue
        A1 = b3[:-len(' and ' + B0)]; B1 = b2[len(A0) + 5:]
        if b4 == A1 + ' and ' + B1: return A0, A1
    return None

# ---------------------------------------------------------------- det-counterfactual diamondcut
def detcf_diamondcut(r):
    rs = r['reasoning']; pr = r['prompt']
    v1 = int(re.search(r'V1 = (\d)', rs).group(1)); x = int(re.search(r'Y_\{X=(\d)\}', rs).group(1))
    op = 'and' if 'Y = X and V3' in rs else 'or'
    mech = re.search(r'We know that (.*?) causes (.*?)\. ', pr).group(2)
    parts = mech.split(' and ', 1)
    negV3 = parts[1].startswith('not ')
    v3 = (1 - v1) if negV3 else v1
    y = (x and v3) if op == 'and' else (x or v3)
    q = pr.rsplit('Would ', 1)[1]
    qneg = ' is not ' in q.split(' if ')[0] or ' not ' in q.split(' if ')[0].split(' is ', 1)[0]
    lab = 'yes' if (y == 0) == qneg else 'no'
    lines = rs.split('\n')
    lines = [('V3 = not V1' if negV3 else 'V3 = V1') if re.match(r'^V3 = (not )?X$', l) else l for l in lines]
    # step5 'Y = y = x op v3' and end
    for i, l in enumerate(lines):
        if re.match(r'^Y = \d = \d (and|or) \d$', l):
            lines[i] = f'Y = {int(y)} = {x} {op} {v3}'
    lines[-1] = str(int(y))
    return lab, '\n'.join(lines)

# ---------------------------------------------------------------- backadj reasoning
def backadj_reasoning(q):
    mt = q['meta']; rs = q.get('reasoning') or {}
    s1, s2 = mt['given_info']
    fmt = lambda s: '{' + ', '.join(s) + '}' if s else 'the empty set'
    bad = mt['bad_candidate_set']; good = s1 if sorted(s2) == sorted(bad) else s2
    m_good = 1 if good is s1 else 2
    return None, (f'Method 1 adjusts for {fmt(s1)}; Method 2 adjusts for {fmt(s2)}.\n'
                  f'{fmt(good)} is a valid backdoor adjustment set for Y given X; {fmt(bad)} is not.\n'
                  f'So Method {m_good} is more correct.')

# ---------------------------------------------------------------- main pipeline
def fix_file(rows, qj, name, changelog, detcf_ids):
    out = []; excluded = []; variant_missing = []; variant_1dp = []
    for r in rows:
        q = qj[int(r['id'])]; mt = q['meta']; qt = mt['query_type']; mid = mt['model_id']; graph = mt['graph_id']
        assert q['question_id'] == int(r['id']) and qt == r['query_type']
        o = dict(r)
        old_prompt, old_label, old_reason = r['prompt'], r['label'], r['reasoning']
        # fix 8: test files lack the question sentence
        if '?' not in o['prompt']:
            o['prompt'] = o['prompt'].rstrip() + ' ' + q['question']
            changelog.append((name, r['id'], 'F8_add_question', 'prompt', '', q['question']))
        flags = dict(ill_posed=False, identifiable_from_prompt=True, rounding_ambiguous='', rounding_ambiguous_1dp='',
                     iv_wald_gives_late_not_ate=False, numbers_rerendered=False, reasoning_regenerated=False,
                     label_flipped=False, old_label=old_label, exact_value='', reasoning_end_matches_label='')
        gi_text = q['given_info']
        if qt in NUMERIC:
            n, _ = world(mid)
            Gs = mt['given_info']; keys = list(Gs)
            Ge = {kk: exact_key(n, kk) for kk in keys}
            oldv = sum((flat(Gs[kk]) for kk in keys), []); newv = sum((flat(Ge[kk]) for kk in keys), [])
            # fix 1a: re-render printed percentages from exact values
            new_gi = rerender_gi(gi_text, oldv, newv, '.0%')
            assert new_gi is not None, (name, r['id'])
            if new_gi != gi_text:
                assert o['prompt'].count(gi_text) == 1
                o['prompt'] = o['prompt'].replace(gi_text, new_gi)
                flags['numbers_rerendered'] = True
                changelog.append((name, r['id'], 'F1_exact_numbers', 'prompt.given_info', gi_text, new_gi))
            # fix 1b: exact label with CLadder's rule
            gt = exact_estimand(qt, mid, mt)
            flags['exact_value'] = f'{gt:.6f}'
            assert answer(qt, mt['groundtruth'], mt) == old_label, ('rule', name, r['id'])
            lab = answer(qt, gt, mt)
            if lab != old_label:
                o['label'] = lab; flags['label_flipped'] = True
                changelog.append((name, r['id'], 'F1_exact_label', 'label', old_label, lab))
            # identification check: formula at exact printed quantities == exact estimand
            step3, e_exact = F(qt, graph, Ge, repr)
            ident = e_exact is not None and abs(ev(e_exact) - gt) < 1e-9
            if qt == 'exp_away' and e_exact is not None:  # answer depends on sign only (no cut)
                ident = (ev(e_exact) > 0) == (gt > 0) and ev(e_exact) != 0
            flags['identifiable_from_prompt'] = ident
            if qt in ('nde', 'nie') and graph == 'arrowhead':
                flags['ill_posed'] = True
            if qt == 'ate' and graph == 'IV':
                flags['iv_wald_gives_late_not_ate'] = True
            # printed quantities (integer %), kind-3 box
            Gp = gi_numbers(Ge, keys, '.0%')
            if e_exact is not None:
                ans, _ = box_answers(qt, graph, Gp, keys, 0.005, mt)
                flags['rounding_ambiguous'] = len(ans - {lab}) > 0
                G1 = gi_numbers(Ge, keys, '.1%')
                ans1, _ = box_answers(qt, graph, G1, keys, 0.0005, mt)
                flags['rounding_ambiguous_1dp'] = len(ans1 - {lab}) > 0
            # fix 5: regenerate step3-5 + end from printed numbers
            lines = (o['reasoning'] or '').split('\n')
            st4 = step4_text(Gp, keys, 2)
            if e_exact is not None:
                _, e2 = F(qt, graph, Gp, lambda v: f'{v:.2f}')
                try:
                    res = ev(e2)
                    step5 = f'{e2} = {res:.2f}'; end = end_text(qt, res, 2)
                    flags['reasoning_end_matches_label'] = answer(qt, res, mt) == lab
                except ZeroDivisionError:
                    step5 = f'{e2} is undefined (a printed probability is 0%)'; end = 'undefined'
                    flags['reasoning_end_matches_label'] = False
            else:
                step5 = 'The given information does not determine this quantity (see identifiable_from_prompt).'; end = 'not identified'
            new_reason = '\n'.join(lines[:3] + [step3, st4, step5, end])
            if new_reason != o['reasoning']:
                o['reasoning'] = new_reason; flags['reasoning_regenerated'] = True
                changelog.append((name, r['id'], 'F5_reasoning', 'reasoning', old_reason, new_reason))
            # variant 1dp (fix 4)
            v1 = dict(o)
            gi1 = rerender_gi(gi_text, oldv, newv, '.1%')
            v1['prompt'] = o['prompt'].replace(new_gi, gi1) if new_gi != gi1 else o['prompt']
            if e_exact is not None:
                G1 = gi_numbers(Ge, keys, '.1%')
                _, e3 = F(qt, graph, G1, lambda v: f'{v:.3f}')
                try:
                    res3 = ev(e3); s5, en = f'{e3} = {res3:.3f}', end_text(qt, res3, 3)
                except ZeroDivisionError:
                    s5, en = f'{e3} is undefined (a printed probability is 0%)', 'undefined'
                v1['reasoning'] = '\n'.join(lines[:3] + [step3, step4_text(G1, keys, 3), s5, en])
            variant_1dp.append({**v1, **flags})
            # fix 3 variant: confounding ETT with P(X|V1) printed
            if qt == 'ett' and graph == 'confounding':
                ph = v1_phrases(new_gi)
                assert ph, r['id']
                X1 = META[mid]['variable_mapping']['X1']
                px = exact_key(n, 'p(X | V1)')
                sent = f' For {ph[0]}, the probability of {X1} is {px[0]:.0%}. For {ph[1]}, the probability of {X1} is {px[1]:.0%}.'
                Gx = dict(Ge); Gx['p(X | V1)'] = px
                kx = keys + ['p(X | V1)']
                s3, ex2 = F(qt, graph, Gx, repr)
                ok = abs(ev(ex2) - gt) < 1e-9
                Gxp = gi_numbers(Gx, kx, '.0%')
                ansx, _ = box_answers(qt, graph, Gxp, kx, 0.005, mt)
                _, ex3 = F(qt, graph, Gxp, lambda v: f'{v:.2f}')
                resx = ev(ex3)
                vm = dict(o); vm['prompt'] = o['prompt'].replace(new_gi, new_gi + sent)
                vm['reasoning'] = '\n'.join(lines[:3] + [s3, step4_text(Gxp, kx, 2), f'{ex3} = {resx:.2f}', end_text(qt, resx, 2)])
                variant_missing.append({**vm, **flags, 'identifiable_from_prompt': ok,
                                        'rounding_ambiguous': len(ansx - {lab}) > 0,
                                        'reasoning_end_matches_label': answer(qt, resx, mt) == lab,
                                        'added_sentence': sent.strip()})
        elif qt == 'det-counterfactual' and graph == 'diamondcut':
            lab, rs = detcf_diamondcut(o)
            if rs != o['reasoning']:
                changelog.append((name, r['id'], 'F6_detcf_reasoning', 'reasoning', o['reasoning'], rs))
                o['reasoning'] = rs; flags['reasoning_regenerated'] = True
            if lab != old_label:
                o['label'] = lab; flags['label_flipped'] = True
                detcf_ids.add(r['id'])
                changelog.append((name, r['id'], 'F6_detcf_label', 'label', old_label, lab))
        elif qt == 'backadj':
            _, rs = backadj_reasoning(q)
            steps = (o['reasoning'] or '').split('\n')
            # the source stores empty steps as the literal string 'nan'; keep only real text
            lines = [l for l in steps if l.strip() and l.strip().lower() not in ('nan', 'none')][:2]
            nr = '\n'.join(lines + ['[backdoor adjustment set for Y given X]', rs, old_label])
            changelog.append((name, r['id'], 'F5_backadj_reasoning', 'reasoning', o['reasoning'], nr))
            o['reasoning'] = nr; flags['reasoning_regenerated'] = True
        o.update(flags)
        (excluded if flags['ill_posed'] else out).append(o)
    # fix 9: duplicates (after all fixes)
    allrows = out + excluded
    first = {}; cnt = collections.Counter(x['prompt'] for x in allrows)
    for x in allrows:
        first.setdefault(x['prompt'], x['id'])
        x['duplicate_of'] = '' if first[x['prompt']] == x['id'] else first[x['prompt']]
        x['n_identical_prompts'] = cnt[x['prompt']]
    for vr in (variant_1dp, variant_missing):
        for x in vr:
            x['duplicate_of'] = ''; x['n_identical_prompts'] = ''
    return out, excluded, variant_1dp, variant_missing

def write(path, rows, extra=()):
    if not rows: return
    cols = list(rows[0].keys())
    for e in extra:
        if e not in cols: cols.append(e)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    changelog = []
    jobs = [('full_v1.5_default', 'full_v1.5_default.csv', 'balanced')] + \
           [(f'test-{s}-v1.5', f'test-{s}-v1.5.csv', s) for s in
            ('anticommonsense', 'balanced', 'commonsense', 'easy', 'hard', 'noncommonsense')]
    summary = {}
    for name, fn, js in jobs:
        rows = list(csv.DictReader(open(f'{DL}/hf/data/{fn}')))
        qj = {x['question_id']: x for x in json.load(open(f'{DL}/v1zip/cladder-v1-q-{js}.json'))}
        detcf_ids = set()
        out, exc, v1dp, vmiss = fix_file(rows, qj, name, changelog, detcf_ids)
        write(f'{OUT}/{name}_fixed.csv', out)
        write(f'{OUT}/{name}_excluded_ill_posed.csv', exc)
        if name == 'full_v1.5_default':
            write(f'{OUT}/{name}_variant_1decimal.csv', v1dp)
            write(f'{OUT}/{name}_variant_confounding_ett_with_PXgivenV1.csv', vmiss)
            ref = set(x['id'] for x in csv.DictReader(open(DETCF_LIST)) if x['query_type'] == 'det-counterfactual')
            summary['detcf_flips'] = len(detcf_ids); summary['detcf_equal_ref37'] = detcf_ids == ref
            summary['detcf_symdiff'] = sorted(detcf_ids ^ ref)
        summary[name] = dict(rows=len(rows), kept=len(out), excluded=len(exc))
        print(name, summary[name], flush=True)
    with open(f'{OUT}/CHANGELOG.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['file', 'id', 'fix', 'field', 'old', 'new']); w.writerows(changelog)
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checks', 'build_summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items()}, indent=1))
