"""Independent exact re-derivation of CLadder v1.5 groundtruths by joint enumeration.
Pure python, no CLadder code imported.  Usage: python3 -I exact.py
Writes worlds_exact.json (per-model exact quantities) and flips.csv.
"""
import json, csv, itertools, re, sys, os, collections
csv.field_size_limit(10**9)
DL = os.environ.get('CLADDER_DL', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
OUT = os.environ.get('CLADDER_VERIFY_OUT', '.')
META = f'{DL}/v1zip/cladder-v1-meta-models.json'
QB = f'{DL}/v1zip/cladder-v1-q-balanced.json'
FULL = f'{DL}/hf/data/full_v1.5_default.csv'


def parse(m):
    cpt = {}  # var -> (parents, nested table of P(var=1|parents))
    for k, v in m['params'].items():
        mm = re.match(r'p\((\w+)(?: \| (.*))?\)', k)
        var = mm.group(1)
        par = [p.strip() for p in mm.group(2).split(',')] if mm.group(2) else []
        cpt[var] = (par, v)
    return cpt


def p1(cpt, var, st):
    par, t = cpt[var]
    for p in par:
        t = t[st[p]]
    while isinstance(t, list):  # root params of 'random' builder are stored as [p]
        assert len(t) == 1
        t = t[0]
    return float(t)


class Net:
    def __init__(self, cpt):
        self.cpt = cpt
        self.vars = list(cpt)

    def joint(self, soft=None):
        """yield (state, prob). soft: var -> P(var=1), replaces mechanism by an independent root."""
        soft = soft or {}
        for bits in itertools.product([0, 1], repeat=len(self.vars)):
            st = dict(zip(self.vars, bits))
            pr = 1.0
            for v in self.vars:
                q = soft[v] if v in soft else p1(self.cpt, v, st)
                pr *= q if st[v] else 1 - q
            yield st, pr

    def marg(self, soft=None, cond=None):
        cond = cond or {}
        tot = 0.0
        acc = collections.defaultdict(float)
        for st, pr in self.joint(soft):
            if any(st[k] != v for k, v in cond.items()):
                continue
            tot += pr
            for v in self.vars:
                if st[v]:
                    acc[v] += pr
        return {v: acc[v] / tot for v in self.vars}

    def parents(self, v):
        return self.cpt[v][0]

    def children(self, v):
        return [w for w in self.vars if v in self.parents(w)]

    def sources(self):
        return [v for v in self.vars if not self.parents(v)]

    def descendants(self, v):
        out, todo = set(), [v]
        while todo:
            for c in self.children(todo.pop()):
                if c not in out:
                    out.add(c); todo.append(c)
        return out

    def mediators(self, x, y):
        # nodes on a directed path x -> ... -> y (excluding x,y)
        anc_y = set(w for w in self.vars if y in self.descendants(w))
        return [w for w in self.vars if w in self.descendants(x) and w in anc_y]


def quantities(m):
    cpt = parse(m)
    n = Net(cpt)
    X, Y = 'X', 'Y'
    obs = n.marg()
    o1 = n.marg(cond={X: 1}); o0 = n.marg(cond={X: 0})
    d1 = n.marg(soft={X: 1.0}); d0 = n.marg(soft={X: 0.0})
    ate = d1[Y] - d0[Y]
    # CLadder-recipe ETT: abduce source marginals given X=1, re-inject independently, do(X=0)
    srcs = {s: o1[s] for s in n.sources() if s != X}
    cf = n.marg(soft={X: 0.0, **srcs})
    ett_recipe = o1[Y] - cf[Y]
    srcs0 = {s: o0[s] for s in n.sources() if s != X}
    ett0_recipe = o0[Y] - n.marg(soft={X: 1.0, **srcs0})[Y]  # CLadder ett(treated=False)
    # textbook ETT via backdoor on all non-descendants of X that are parents of X (adjust on pa(X))
    # E[Y_x0 | X=1] = sum_z P(z|X=1) P(Y|do x0, z) with z = parents of X (valid when pa(X) observed in model)
    pa = n.parents(X)
    if pa:
        tot = 0.0
        for zb in itertools.product([0, 1], repeat=len(pa)):
            z = dict(zip(pa, zb))
            pz = 1.0
            # P(z | X=1)
            jz = sum(pr for st, pr in n.joint() if st[X] == 1 and all(st[k] == v for k, v in z.items()))
            jx = sum(pr for st, pr in n.joint() if st[X] == 1)
            # P(Y=1 | do(X=0), z): z are non-descendants so condition in mutilated graph
            yz = n.marg(soft={X: 0.0}, cond=z)[Y]
            tot += jz / jx * yz
        ett_text = o1[Y] - tot
    else:
        ett_text = ate
    med = n.mediators(X, Y)
    res = dict(ETT0_recipe=ett0_recipe, obs=obs, PY1_X1=o1[Y], PY1_X0=o0[Y], ATE=ate, ETT_recipe=ett_recipe, ETT_text=ett_text,
               mediators=med)
    if not med:
        res.update(NDE_recipe=ate, NIE_recipe=0.0 if False else None, NDE_text=ate, NIE_text=0.0)
    if med:
        mx0 = {v: d0[v] for v in med}; mx1 = {v: d1[v] for v in med}
        res['NDE_recipe'] = n.marg(soft={X: 1.0, **mx0})[Y] - d0[Y]
        res['NIE_recipe'] = n.marg(soft={X: 0.0, **mx1})[Y] - d0[Y]
        # textbook: E[Y(x, M(x'))] with nested counterfactual, cross-world independence of mechanisms
        # (i.e. mediation formula): enumerate pre-treatment/non-mediator vars W, mediators M.
        res['NDE_text'] = nested(n, 1, 0) - nested(n, 0, 0)
        res['NIE_text'] = nested(n, 0, 1) - nested(n, 0, 0)
    else:
        res['NIE_recipe'] = 0.0
    return res


def nested(n, xy, xm):
    """E[Y(X=xy, M(X=xm))] for mediators M = descendants of X that are ancestors of Y.
    Non-mediator vars keep a single world; mediators follow their mechanism with X=xm and
    mediator parents from the xm world; Y uses X=xy and mediators from the xm world.
    Assumes (as in all CLadder graphs) independent exogenous noise per variable."""
    X, Y = 'X', 'Y'
    med = n.mediators(X, Y)
    others = [v for v in n.vars if v not in med and v not in (X, Y)]
    # topological order
    order = []
    def visit(v):
        if v in order: return
        for p in n.parents(v): visit(p)
        order.append(v)
    for v in n.vars: visit(v)
    tot = 0.0
    free = [v for v in order if v not in (X, Y) and (v in med or v in others)]
    # only variables that are non-descendants or mediators matter for Y; others that are descendants of Y ignored
    free = [v for v in free if Y not in n.parents(v) and v not in n.descendants(Y)]
    for bits in itertools.product([0, 1], repeat=len(free)):
        st = dict(zip(free, bits))
        pr = 1.0
        for v in free:
            ctx = dict(st)
            if X in n.parents(v):
                ctx[X] = xm if v in med else None
                if ctx[X] is None:
                    raise ValueError('non-mediator child of X ' + v)
            q = p1(n.cpt, v, ctx)
            pr *= q if st[v] else 1 - q
        ctx = dict(st); ctx[X] = xy
        tot += pr * p1(n.cpt, Y, ctx)
    return tot


def cladder_answer(qt, gt, polarity, treated):
    """Re-implementation of CLadder's yes/no rule (queries/*.py)."""
    if qt == 'marginal':
        s = 2 * gt - 1
        sign = 0 if s == 0 else (1 if s > 0 else -1)
        if sign == 0 or abs(gt) < 0.005:
            a = -1
        else:
            a = sign * (-1) ** ((1 - int(polarity)) + (1 - int(treated)))
    elif qt == 'ate':
        sign = 0 if gt == 0 else (1 if gt > 0 else -1)
        if sign == 0: return 'unknown'
        a = -1 if abs(gt) < 0.005 else sign * (-1) ** ((1 - 1) + (1 - int(polarity)))
    else:  # ett, nde, nie
        sign = 0 if gt == 0 else (1 if gt > 0 else -1)
        extra = 1 if qt == 'ett' else 0  # ett.py: (-1)**(1 + 1-result + 1-polarity)
        if sign == 0 and qt == 'ett': return 'unknown'
        a = -1 if (sign == 0 or abs(gt) < 0.005) else sign * (-1) ** (extra + 1 - int(polarity))
    return {-1: 'no', 1: 'yes', 0: 'unknown'}[a]


