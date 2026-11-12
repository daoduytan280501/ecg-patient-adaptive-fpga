"""E13: chon head_scratch_fair tren DS1 (S_rob cao nhat trong cau hinh on dinh; lua chon phu: mF1 sach cao nhat).
Lua chon phu cung tinh cho GA-R tu bang chon rev5b. Ghi e13_selected.json (bam SHA-256 truoc khi danh gia DS2 / kiem tra mu)."""
import json, pandas as pd
OUT = '../out/bosung_20261026'
H = pd.concat([pd.read_csv(f'{OUT}/ds1_e13_part{i}.csv') for i in (0, 1)]).reset_index(drop=True)
assert len(H) == 216 and H.hp.nunique() == 216, len(H)
H.to_csv(f'{OUT}/ds1_e13_head_scratch_fair.csv', index=False)
G = pd.read_csv('../out/ds1_selection_rev5b_20261025.csv'); G = G[G.method == 'GA-R'].reset_index(drop=True)
def best(d, col):
    s = d[d.stable]; r = s.loc[s[col].idxmax()]
    return dict(hp=json.loads(r.hp), **{k: (float(r[k]) if k not in ('n_unstable', 'stable') else (int(r[k]) if k == 'n_unstable' else bool(r[k])))
                                        for k in ['mF1_clean', 'mF1_uniform', 'mF1_pair', 'S_rob', 'n_unstable', 'stable']})
out = dict(S_rob=best(H, 'S_rob'), clean=best(H, 'mF1_clean'), GA_R_S_rob=best(G, 'S_rob'), **{'GA-R_clean': best(G, 'mF1_clean')},
           n_configs=len(H), n_unstable=int((~H.stable).sum()), rule='protocol_bosung_20261026.md E13')
assert out['GA_R_S_rob']['hp'] == dict(beta=0.1, cap=1.0, trust=0.5, lr=0.05)
json.dump(out, open(f'{OUT}/e13_selected.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
print(H.sort_values('S_rob', ascending=False).head(10)[['hp', 'mF1_clean', 'mF1_uniform', 'mF1_pair', 'S_rob', 'n_unstable']].to_string())
