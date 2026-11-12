"""Figures for the BSPC manuscript (English labels, Times-compatible font, vector PDF).
All numbers come from the locked result files of the study (no value is typed by hand)."""
import sys, json, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
import wfdb

for f in font_manager.findSystemFonts():
    if 'texgyretermes' in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'TeX Gyre Termes'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.4, 'axes.spines.top': True, 'axes.spines.right': True,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': '0.3', 'legend.framealpha': 1.0})
# panel captions (a), (b), ... are placed below the x-axis label instead of above the axes (supervisor comment 11/11/2026)
from matplotlib.axes import Axes
_st, _sx = Axes.set_title, Axes.set_xlabel
def _set_title_below(self, t, *a, **k):
    self._sub = t; _sx(self, (getattr(self, '_xl', '') + '\n' if getattr(self, '_xl', '') else '') + t)
def _set_xlabel(self, l, *a, **k):
    self._xl = l; return _sx(self, l + ('\n' + self._sub if getattr(self, '_sub', None) else ''), *a, **k)
Axes.set_title = _set_title_below; Axes.set_xlabel = _set_xlabel
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM          # Elsevier single, 1.5 and double column
OUT = '/home/claude/bspc/figs/'
R5 = '/home/claude/planA_rev5/'
P = R5 + 'out/paper_rev5b/'; Q = R5 + 'out/bosung_20261026/'
EV = '/mnt/user-data/uploads/ECG_FPGA_1DCNN/planA/06_eval/'
COL = {'none': '#000000', 'offset_int': '#0072B2', 'GA-R': '#D55E00', 'head_scratch_fair': '#009E73',
       'head_scratch': '#999999', 'gradient_anchor': '#CC79A7', 'head_global': '#E69F00'}
LAB = {'none': 'none', 'offset_int': 'offset_int', 'GA-R': 'GA-R', 'head_scratch_fair': 'head_scratch (fair)',
       'head_scratch': 'head_scratch (original)', 'gradient_anchor': 'gradient_anchor', 'head_global': 'head_global'}
MK = {'none': 'x', 'offset_int': 'D', 'GA-R': 'o', 'head_scratch_fair': 's', 'head_scratch': 'v',
      'gradient_anchor': '^', 'head_global': 'P'}
NUM = {}


def save(fig, name):
    fig.savefig(OUT + name + '.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT + name + '.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig); print('saved', name)


# ------------------------------------------------------------------ Fig: example beats
def fig_beats():
    D = '/mnt/user-data/uploads/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0/'
    MAP = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
           'V': 'V', 'E': 'V', 'F': 'F'}
    pick = [('N', 100, 'N'), ('S', 232, 'S'), ('V', 119, 'V'), ('F', 208, 'F')]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 38 * MM), sharey=True)
    t = (np.arange(360) - 162) / 180
    for k, (cls, rec, want) in enumerate(pick):
        r = wfdb.rdrecord(D + str(rec)); a = wfdb.rdann(D + str(rec), 'atr')
        x = r.p_signal[:, r.sig_name.index('MLII')].astype(float)
        b = median_filter(median_filter(x, 73, mode='nearest'), 217, mode='nearest'); x = x - b
        bb, aa = butter(4, 40 / 180, 'low'); x = filtfilt(bb, aa, x)[::2]
        idx = [s // 2 for s, sym in zip(a.sample, a.symbol) if MAP.get(sym) == want]
        idx = [i for i in idx if i - 162 >= 0 and i + 198 <= len(x)]
        sel = [idx[len(idx) // 5], idx[len(idx) // 2], idx[4 * len(idx) // 5]]
        for j, i in enumerate(sel):
            w = x[i - 162:i + 198]; w = w - np.median(w); w = w / np.abs(w).max()
            ax[k].plot(t, w, lw=0.8, color=['#0072B2', '#D55E00', '#009E73'][j])
        ax[k].axvline(0, color='k', ls=':', lw=0.6)
        ax[k].set_title(f'({"abcd"[k]}) Class {cls} (record {rec})'); ax[k].set_xlabel('Time from R peak (s)')
        ax[k].set_xlim(-0.9, 1.1)
    ax[0].set_ylabel('Normalised amplitude')
    fig.tight_layout(w_pad=0.6); save(fig, 'fig_beats')


# ------------------------------------------------------------------ Fig: step-size stability (record 213)
def fig_eta():
    sys.path.insert(0, R5 + 'src')
    import rev4_data as rd
    D2 = pickle.load(open('/home/claude/planA_rev4/out/D2.pkl', 'rb'))
    y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']; NC = 4

    def traj(E, yc, W0, b0, A, lr, Eev, beta=.1, cap=1., trust=.5, lam=.01, T=300):
        Aa, ya = A; Eall = np.vstack([E, Aa]); yall = np.concatenate([yc, ya])
        cnt = np.bincount(yc, minlength=NC).astype(float)
        cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.)
        w = cw[yc]; w = (1 - beta) * w / w.sum(); ca = np.bincount(ya, minlength=NC).astype(float)
        sw = np.concatenate([w, beta / (NC * ca[ya])]); s = 1 / (Eall.std() + 1e-9); Es = Eall * s
        Wt0 = W0 / s; Wt = Wt0.copy(); b = b0.copy(); Y = np.eye(NC)[yall]; rl = trust * np.linalg.norm(Wt0)
        L, ch, prev = [], [], None
        for _ in range(T):
            Z = Es @ Wt + b; Z -= Z.max(1, keepdims=True); Pm = np.exp(Z); Pm /= Pm.sum(1, keepdims=True)
            L.append(float(-(sw * np.log(Pm[np.arange(len(yall)), yall] + 1e-300)).sum()))
            G = (Pm - Y) * sw[:, None]; Wt -= lr * (Es.T @ G + lam * (Wt - Wt0)); b -= lr * (G.sum(0) + lam * (b - b0))
            Dd = Wt - Wt0; n = np.linalg.norm(Dd)
            if n > rl: Wt = Wt0 + Dd * (rl / n)
            pr = (Eev @ (Wt * s) + b).argmax(1)
            ch.append(np.nan if prev is None else int((pr != prev).sum())); prev = pr
        return np.array(L), np.array(ch)
    S = D2['seeds'][3]; m = rec == 213
    res = {lr: traj(S['E'][m & cal], y[m & cal], S['W0'], S['b0'], (S['A'], rd.ANC_Y), lr, S['E'][m & ev])
           for lr in (0.5, 0.05)}
    NUM['rec213'] = dict(n_cal=int((m & cal).sum()), n_ev=int((m & ev).sum()),
                         changes_last50={str(k): float(np.nansum(v[1][-50:])) for k, v in res.items()})
    fig, ax = plt.subplots(1, 2, figsize=(W2, 50 * MM)); it = np.arange(1, 301)
    ax[0].plot(it, res[0.5][0], color=COL['GA-R'], lw=0.6, label=r'$\eta=0.5$')
    ax[0].plot(it, res[0.05][0], color=COL['offset_int'], lw=1.2, label=r'$\eta=0.05$ (selected)')
    ax[0].set_xlabel('Iteration'); ax[0].set_ylabel('Training loss $\\mathcal{L}$'); ax[0].legend(loc='upper right')
    ax[0].set_title('(a) Loss after each gradient step')
    ax[1].semilogy(it, np.maximum(res[0.5][1], .5), color=COL['GA-R'], lw=0.6, label=r'$\eta=0.5$')
    ax[1].semilogy(it, np.maximum(res[0.05][1], .5), color=COL['offset_int'], lw=1.2, label=r'$\eta=0.05$ (selected)')
    ax[1].set_xlabel('Iteration'); ax[1].set_ylabel('Labels changed vs. previous step')
    ax[1].set_title(f'(b) Predicted labels of the {int((m & ev).sum())} evaluation beats'); ax[1].legend(loc='center right')
    fig.tight_layout(); save(fig, 'fig_eta')


# ------------------------------------------------------------------ Fig: confusion matrices (board, seed 3)
def fig_cm():
    C = json.load(open(R5 + 'report/cm_seed3_rev5b.json')); cls = ['N', 'S', 'V', 'F']
    fig, ax = plt.subplots(1, 3, figsize=(W2, 56 * MM))
    for k, mth in enumerate(['none', 'offset_int', 'GA-R']):
        M = np.array(C[mth], float); Rn = M / M.sum(1, keepdims=True)
        ax[k].imshow(Rn, cmap='Blues', vmin=0, vmax=1); ax[k].grid(False)
        for i in range(4):
            for j in range(4):
                ax[k].text(j, i, f'{int(M[i, j])}', ha='center', va='center', fontsize=7,
                           color='white' if Rn[i, j] > 0.6 else 'black')
        ax[k].set_xticks(range(4)); ax[k].set_yticks(range(4)); ax[k].set_xticklabels(cls); ax[k].set_yticklabels(cls)
        ax[k].set_xlabel('Predicted class'); ax[k].set_title(f'({"abc"[k]}) {LAB[mth]}')
    ax[0].set_ylabel('True class')
    fig.tight_layout(); save(fig, 'fig_cm')


# ------------------------------------------------------------------ Fig: per-seed jk
def fig_seed():
    A = pd.read_csv(P + 'main_rows.csv'); B = pd.read_csv(Q + 'e13_main_rows.csv')
    B = B[B.method == 'head_scratch_fair']; A = pd.concat([A, B])
    order = ['none', 'head_scratch', 'head_scratch_fair', 'head_global', 'gradient_anchor', 'offset_int', 'GA-R']
    fig, ax = plt.subplots(figsize=(W1, 55 * MM))
    for k, mth in enumerate(order):
        v = A[A.method == mth].sort_values('seed').jk.values
        ax.scatter(np.full(5, k) + np.linspace(-.12, .12, 5), v, s=10, color=COL[mth], marker=MK[mth], zorder=3)
        ax.hlines(v.mean(), k - .25, k + .25, color='k', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['none', 'HS (orig.)', 'HS (fair)', 'HG', 'GA', 'offset_int', 'GA-R'], rotation=35, ha='right')
    ax.set_ylabel(r'$j\kappa$ (DS2 evaluation segment)'); ax.set_ylim(0.6, 0.92)
    fig.tight_layout(); save(fig, 'fig_seed')


# ------------------------------------------------------------------ Fig: label budget
def fig_budget():
    B = pd.read_csv(P + 'budget.csv'); g = B.groupby(['method', 'budget'])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 50 * MM))
    for mth in ['offset_int', 'gradient_anchor', 'GA-R']:
        for k, col in enumerate(['jk', 'Se_S']):
            mu = g[col].mean().loc[mth]; sd = g[col].std().loc[mth]
            ax[k].errorbar(mu.index, mu.values, yerr=sd.values, color=COL[mth], marker=MK[mth], ms=3.5,
                           capsize=2, lw=1, label=LAB[mth])
    ax[0].axhline(0.645, color='k', ls=':', lw=0.8); ax[0].text(5, 0.652, 'none', ha='right', fontsize=7)
    ax[1].axhline(0.305, color='k', ls=':', lw=0.8); ax[1].text(5, 0.315, 'none', ha='right', fontsize=7)
    for k, (yl, tt) in enumerate([(r'$j\kappa$', '(a) $j\\kappa$'), (r'Se$_S$', '(b) S-class sensitivity')]):
        ax[k].set_xlabel('Labelled calibration time (min)'); ax[k].set_ylabel(yl); ax[k].set_title(tt); ax[k].set_xticks([1, 2, 3, 4, 5])
    ax[0].legend(loc='center right'); fig.tight_layout(); save(fig, 'fig_budget')


# ------------------------------------------------------------------ Fig: label noise
def fig_noise():
    A = pd.read_csv(P + 'noise.csv'); B = pd.read_csv(Q + 'e13_noise.csv'); B = B[B.method == 'head_scratch_fair']
    N = pd.concat([A, B]); M = pd.read_csv(P + 'main_rows.csv'); M13 = pd.read_csv(Q + 'e13_main_rows.csv')
    clean = dict(M.groupby('method').jk.mean()); clean['head_scratch_fair'] = M13[M13.method == 'head_scratch_fair'].jk.mean()
    meth = ['offset_int', 'GA-R', 'head_scratch_fair', 'gradient_anchor', 'head_global', 'head_scratch']
    fig, ax = plt.subplots(1, 3, figsize=(W2, 66 * MM), sharey=True)
    NUM['noise_mean'] = {}
    for k, (kind, tt) in enumerate([('miss', '(a) Missed ectopic beats'), ('pair', '(b) Pairwise confusion'),
                                     ('uniform', '(c) Uniform relabelling')]):
        for mth in meth:
            g = N[(N.method == mth) & (N.kind == kind)].groupby('p').jk.mean()
            xs = [0] + [100 * p for p in g.index]; ys = [clean[mth]] + list(g.values)
            ax[k].plot(xs, ys, color=COL[mth], marker=MK[mth], ms=3.5, lw=1.4 if mth in ('GA-R', 'offset_int', 'head_scratch_fair') else 0.8,
                       ls='-' if mth in ('GA-R', 'offset_int', 'head_scratch_fair') else '--', label=LAB[mth])
        ax[k].set_title(tt); ax[k].set_xlabel('Noise rate (%)'); ax[k].set_xticks([0, 5, 10, 20])
    for mth in meth:
        NUM['noise_mean'][mth] = float(N[N.method == mth].groupby(['kind', 'p']).jk.mean().mean())
    ax[0].set_ylabel(r'$j\kappa$'); ax[0].set_ylim(0.2, 0.92)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.84)); save(fig, 'fig_noise')


# ------------------------------------------------------------------ Fig: per-record change, DS2 + external
def recdelta_ds2():
    sys.path.insert(0, R5 + 'src')
    import rev4_methods as Mth, rev4_data as rd, pa_adapt as pa, fair_methods  # noqa
    D2 = pickle.load(open('/home/claude/planA_rev4/out/D2.pkl', 'rb'))
    y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']; RECS = sorted(np.unique(rec))
    SEL5 = json.load(open(R5 + 'out/ds1_selected_rev5b_20261025.json'))['selected']
    SELJ = json.load(open(Q + 'e13_selected.json'))
    cfg = {'none': ('none', {}), 'head_scratch_fair': ('head_scratch_fair', SELJ['S_rob']['hp']),
           'GA-R': ('GA-S', SEL5['GA-R']['hp']), 'offset_int': ('offset_int', SEL5['offset_int']['hp'])}
    out = {}
    for name, (mth, hp) in cfg.items():
        vals = []
        for s in rd.SEEDS:
            Sd = D2['seeds'][s]
            y_, p_, r_ = Mth.run(Sd['E'], y, rec, cal, ev, Sd['W0'], Sd['b0'], (Sd['A'], rd.ANC_Y), mth, fW=Sd['fw'], **hp)
            pr = pa.per_record_scores(y_, p_, r_); vals.append([pr[r]['balacc'] for r in RECS])
        out[name] = np.nanmean(np.array(vals, float), 0)
    return RECS, out


def fig_records():
    RECS, V = recdelta_ds2()
    d = {m: V[m] - V['none'] for m in ['head_scratch_fair', 'GA-R', 'offset_int']}
    NUM['ds2_perrecord'] = {m: dict(harmed=int((v < -0.05).sum()), up=int((v > 0).sum()), down=int((v < 0).sum()),
                                     worst=float(v.min())) for m, v in d.items()}
    fig = plt.figure(figsize=(W2, 112 * MM)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.72, wspace=0.3)
    a0 = fig.add_subplot(gs[0, :]); o = np.argsort(d['GA-R']); x = np.arange(len(RECS)); wd = 0.27
    for k, m in enumerate(['head_scratch_fair', 'GA-R', 'offset_int']):
        a0.bar(x + (k - 1) * wd, d[m][o], wd, color=COL[m], label=LAB[m])
    a0.axhline(-0.05, color='k', ls='--', lw=0.7); a0.text(len(RECS) - 0.5, -0.07, 'harm threshold ($-0.05$)', ha='right', va='top', fontsize=7)
    a0.set_xticks(x); a0.set_xticklabels([str(RECS[i]) for i in o], rotation=90, fontsize=6.5); a0.set_xlim(-.6, len(RECS) - .4)
    a0.set_xlabel('DS2 record'); a0.set_ylabel(r'$\Delta$ balanced accuracy vs. none'); a0.legend(loc='upper left', ncol=3)
    a0.set_title('(a) DS2, clean calibration labels, mean of five seeds')
    files = [('DS2', None), ('INCART', 'external_rev5c_20261029/ext_rev5c_incartdb_per_record.csv'),
             ('SVDB', 'external_rev5c_20261029/ext_rev5c_svdb_per_record.csv'),
             ('European ST-T (blind)', 'blind_edb_20261026/blind_edb_per_record.csv')]
    NUM['harm_all'] = {}
    for k, (db, f) in enumerate(files):
        ax = fig.add_subplot(gs[1, k])
        if f is None: dd = d
        else:
            T = pd.read_csv(EV + f, index_col=0)
            dd = {m: (T[m + '|balacc'] - T['none|balacc']).values for m in d}
        NUM['harm_all'][db] = {m: int((np.asarray(v) < -0.05).sum()) for m, v in dd.items()}
        rng = np.random.default_rng(0)
        for j, m in enumerate(['head_scratch_fair', 'GA-R', 'offset_int']):
            v = np.asarray(dd[m], float); v = v[~np.isnan(v)]
            ax.scatter(j + rng.uniform(-.18, .18, len(v)), v, s=4, color=COL[m], alpha=.7, lw=0)
            ax.boxplot(v, positions=[j], widths=.5, showfliers=False, medianprops=dict(color='k', lw=0.8),
                       boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            ax.text(j, -0.62, f'{int((v < -0.05).sum())}/{len(v)}', ha='center', fontsize=6.5, color=COL[m])
        ax.axhline(-0.05, color='k', ls='--', lw=0.6); ax.axhline(0, color='k', lw=0.4)
        ax.set_ylim(-0.68, 0.55); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['HS (fair)', 'GA-R', 'offset'], fontsize=6.5)
        ax.set_title(f'({"bcde"[k]}) {db}')
        if k == 0: ax.set_ylabel(r'$\Delta$ balanced accuracy vs. none')
    save(fig, 'fig_records')


# ------------------------------------------------------------------ Fig: Pareto
def fig_pareto():
    B5 = json.load(open(R5 + 'board_out/f2a_rev5b_board_20261025.json'))['energy']
    B5c = json.load(open(R5 + 'board_out/f2a_rev5c_board_20261029.json'))['energy']
    # session 2 (29/10) for the four rules measured together; session 1 (25/10) only for the two rules not re-measured
    E = {'offset_int': B5c['offset_int_tau20']['energy_mJ_per_record'], 'GA-R': B5c['GA_R']['energy_mJ_per_record'],
         'head_scratch': B5c['head_scratch']['energy_mJ_per_record'], 'head_scratch_fair': B5c['head_scratch_fair']['energy_mJ_per_record'],
         'gradient_anchor': B5['gradient_anchor']['energy_mJ_per_record'], 'head_global': B5['head_global']['energy_mJ_per_record']}
    S1 = {'gradient_anchor', 'head_global'}
    M = pd.read_csv(P + 'main_rows.csv').groupby('method').jk.mean(); M13 = pd.read_csv(Q + 'e13_main_rows.csv')
    J = dict(M); J['head_scratch_fair'] = M13[M13.method == 'head_scratch_fair'].jk.mean()
    NM = NUM['noise_mean']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    off = {'offset_int': (6, -3), 'GA-R': (6, -8), 'gradient_anchor': (0, -11), 'head_scratch': (-6, -2),
           'head_global': (-5, -8), 'head_scratch_fair': (7, 3)}
    for m in E:
        if m in S1: ax[0].scatter(E[m], J[m], s=26, marker=MK[m], facecolor='white', edgecolor=COL[m], lw=1.0, zorder=3)
        else: ax[0].scatter(E[m], J[m], s=26, marker=MK[m], color=COL[m], zorder=3)
        ax[0].annotate(LAB[m], (E[m], J[m]), textcoords='offset points', xytext=off[m], fontsize=8.5,
                       ha='right' if off[m][0] < 0 else ('center' if off[m][0] == 0 else 'left'))
    ax[0].scatter([53000, 68000], [0.908, 0.844], marker='s', facecolor='none', edgecolor='k', s=22)
    ax[0].annotate('ft_full (est.)', (53000, 0.908), textcoords='offset points', xytext=(-5, -9), fontsize=8.5, ha='right')
    ax[0].annotate('ft_full_anchor (est.)', (68000, 0.844), textcoords='offset points', xytext=(-5, 4), fontsize=8.5, ha='right')
    ax[0].set_xscale('log'); ax[0].set_xlim(0.05, 3e5); ax[0].set_ylim(0.828, 0.915)
    ax[0].set_xlabel('Adaptation energy per record on the Cortex-A53 (mJ)'); ax[0].set_ylabel(r'Clean-label $j\kappa$')
    ax[0].set_title('(a) Cost versus performance')
    ax[0].scatter([], [], s=26, marker='o', color='grey', label='session 2 (same session)')
    ax[0].scatter([], [], s=26, marker='o', facecolor='white', edgecolor='grey', label='session 1 only')
    ax[0].legend(loc='center', bbox_to_anchor=(0.71, 0.42), fontsize=8)
    off2 = {'offset_int': (5, 3), 'GA-R': (-5, -9), 'gradient_anchor': (5, 3), 'head_scratch': (5, 3),
            'head_global': (-5, 5), 'head_scratch_fair': (-5, 4)}
    for m in E:
        ax[1].scatter(J[m], NM[m], s=22, marker=MK[m], color=COL[m], zorder=3)
        ax[1].annotate(LAB[m], (J[m], NM[m]), textcoords='offset points', xytext=off2[m], fontsize=8.5,
                       ha='right' if off2[m][0] < 0 else 'left')
    ax[1].set_xlabel(r'Clean-label $j\kappa$'); ax[1].set_ylabel(r'Mean $j\kappa$ over nine noisy conditions')
    ax[1].set_title('(b) Performance versus robustness'); ax[1].set_xlim(0.835, 0.912); ax[1].set_ylim(0.55, 0.875)
    NUM['pareto'] = dict(E=E, J=J)
    fig.tight_layout(); save(fig, 'fig_pareto')


# ------------------------------------------------------------------ Fig: DS1 vs DS2 selection
def fig_grid():
    S = json.load(open(Q + 'e15c_summary.json'))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    for k, (mth, f, sel) in enumerate([('GA-R', 'e15c_GA-R_ds1_vs_ds2.csv', S['GA-R']),
                                        ('head_scratch_fair', 'e15c_head_scratch_fair_ds1_vs_ds2.csv', S['head_scratch_fair'])]):
        T = pd.read_csv(Q + f); T = T[T.stable == True]
        ax[k].scatter(T.S_rob_ds1, T.S_rob_ds2, s=5, color=COL[mth], alpha=.55, lw=0)
        ax[k].scatter([sel['sel_S_rob_ds1']], [sel['sel_S_rob_ds2']], s=60, marker='*', color='k', zorder=4,
                      label=f'selected on DS1\n(DS2 rank {sel["sel_rank_ds2"]}/{sel["n_stable"]})')
        ax[k].axhline(sel['oracle_S_rob_ds2'], color='k', ls=':', lw=0.7, label='best on DS2')
        ax[k].set_xlabel(r'$S_{\mathrm{rob}}$ on DS1 (selection)'); ax[k].set_ylabel(r'$S_{\mathrm{rob}}$ on DS2 (report only)')
        ax[k].set_title(f'({"ab"[k]}) {LAB[mth]}: Spearman $\\rho_s$ = {sel["spearman"]:.2f}')
        ax[k].legend(loc='lower right', fontsize=8)
    fig.tight_layout(); save(fig, 'fig_grid')


if __name__ == '__main__':
    fig_beats(); fig_eta(); fig_cm(); fig_seed(); fig_budget(); fig_noise(); fig_records()
    big = {'font.size': 9.5, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8}
    with plt.rc_context(big): fig_pareto(); fig_grid()
    json.dump(NUM, open(OUT + 'fig_numbers.json', 'w'), indent=1, default=float)
    print(json.dumps(NUM, indent=1, default=float)[:3000])
