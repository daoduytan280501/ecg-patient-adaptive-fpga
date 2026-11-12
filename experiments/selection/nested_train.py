"""Huan luyen lai model fold DS1 KHONG ro ri (gop y thay 25/10, diem 2).
Fold k: chi dung ban ghi train T_k = DS1 \\ V_k.
 - CV 3 fold ben trong T_k (make_folds tren T_k, 208 luon o train): moi fold trong -> epoch tot nhat
   (checkpoint macro-F1 tren fold trong) + offset (tune_bias tren fold trong).
 - Train lai tren TOAN BO T_k voi epoch = TB cac epoch trong, offset = TB cac offset trong.
"""
import os, sys, json, time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import pc_pre_int as pp, rev4_data as rd

MITDB = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
JOBS = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in sys.argv[3:]]   # seed:fold
NUM_CLASSES, CUR_NRR, SEED0 = 4, 8, 42
LR, MAX_EPOCHS, BATCH, PATIENCE = 5e-4, 50, 512, 10
CFG = dict(conv=[(16, 7), (32, 5), (32, 5)], rr_dense=16, cw_cap=25)
RR_DIV = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0], np.float32)
rr_norm = lambda R: (R[:, :8] / RR_DIV).clip(0, 3.9).astype(np.float32)

def build_model(L):
    bi = layers.Input(shape=(L, 1), name='beat'); x = bi
    for i, (f, k) in enumerate(CFG['conv'], 1):
        x = layers.Conv1D(f, k, padding='same', name=f'conv1d_{i}')(x); x = layers.ReLU(name=f'relu_{i}')(x)
        x = layers.MaxPooling1D(2, name=f'pool_{i}')(x)
    x = layers.Flatten(name='flatten')(x)
    ri = layers.Input(shape=(CUR_NRR,), name='rr')
    r = layers.Dense(CFG['rr_dense'], activation='relu', name='rr_dense')(ri)
    x = layers.Concatenate(name='concat')([x, r])
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    return models.Model([bi, ri], layers.Dense(NUM_CLASSES, name='dense_out')(x))

def class_weights_effective(y, beta=0.999, max_ratio=10.0):
    cnt = np.maximum(np.bincount(y, minlength=NUM_CLASSES), 1)
    w = (1 - beta) / (1 - np.power(beta, cnt)); w = w / w.min(); w = np.minimum(w, max_ratio)
    return w / np.mean(w[np.bincount(y, minlength=NUM_CLASSES) > 0])

def focal_loss(alpha, gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, logits):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32); logp = tf.nn.log_softmax(logits)
        lp = tf.gather_nd(logp, tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)); p = tf.exp(lp)
        return tf.reduce_mean(-tf.gather(alpha, y_true) * tf.pow(1.0 - p, gamma) * lp)
    return loss

def time_stretch(x, max_frac=0.10):
    L = tf.shape(x)[1]; s = tf.random.uniform([], 1.0 - max_frac, 1.0 + max_frac)
    newL = tf.cast(tf.round(tf.cast(L, tf.float32) * s), tf.int32)
    y = tf.image.resize(x[:, :, None, :], [newL, 1], method='bilinear')[:, :, 0, :]; diff = newL - L
    y = tf.cond(diff > 0, lambda: y[:, diff // 2: diff // 2 + L, :],
                lambda: tf.pad(y, [[0, 0], [(-diff) // 2, -diff - (-diff) // 2], [0, 0]], mode='SYMMETRIC'))
    return tf.reshape(y, tf.shape(x))

def augment(x):
    x = time_stretch(x); B = tf.shape(x)[0]; L = tf.shape(x)[1]
    scale = tf.random.uniform([B, 1, 1], 0.85, 1.15); noise = tf.random.normal(tf.shape(x), stddev=0.02)
    t = tf.cast(tf.range(L), tf.float32)[None, :, None] / tf.cast(L, tf.float32)
    wander = tf.random.uniform([B, 1, 1], -0.08, 0.08) * tf.sin(2.0 * np.pi * tf.random.uniform([B, 1, 1], 0.2, 1.0) * t + tf.random.uniform([B, 1, 1], 0, 6.28))
    x = tf.roll(x, tf.random.uniform([], -4, 5, dtype=tf.int32), axis=1)
    return tf.clip_by_value(x * scale + noise + wander, -1.98, 1.98)

def make_ds(X, R, y):
    ds = tf.data.Dataset.from_tensor_slices(({'beat': X[..., None], 'rr': R}, y)).shuffle(len(y), seed=SEED0).batch(BATCH)
    ds = ds.map(lambda f, yy: (dict(beat=augment(f['beat']), rr=f['rr']), yy), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def macro_f1(y, p):
    f = []
    for c in range(NUM_CLASSES):
        if (y == c).sum() == 0: continue
        tp = ((y == c) & (p == c)).sum(); se = tp / (y == c).sum(); pv = tp / max((p == c).sum(), 1)
        f.append(2 * se * pv / (se + pv) if se + pv else 0.0)
    return float(np.mean(f))

def tune_bias(logits, y, iters=3, grid=np.linspace(-3, 3, 61)):
    off = np.zeros(NUM_CLASSES); best = macro_f1(y, logits.argmax(1)); present = set(np.unique(y).tolist())
    for _ in range(iters):
        for c in [c for c in range(1, NUM_CLASSES) if c in present]:
            for g in grid:
                o = off.copy(); o[c] = g; f = macro_f1(y, (logits + o).argmax(1))
                if f > best + 1e-6: best, off = f, o
    return off, best

def train(X, R, y, epochs, val=None, seed=1):
    tf.keras.utils.set_random_seed(seed)
    m = build_model(X.shape[1]); m.compile(tf.keras.optimizers.Adam(LR, use_ema=True, ema_momentum=0.99), loss=focal_loss(class_weights_effective(y, max_ratio=CFG['cw_cap'])))
    ds = make_ds(X, rr_norm(R), y); best = (-1, 0, None); wait = 0
    for ep in range(epochs):
        m.fit(ds, epochs=1, verbose=0)
        if val is not None:
            Xv, Rv, yv = val
            f = macro_f1(yv, m.predict([Xv[..., None], rr_norm(Rv)], batch_size=4096, verbose=0).argmax(1))
            if f > best[0]: best, wait = (f, ep + 1, m.get_weights()), 0
            else:
                wait += 1
                if wait >= PATIENCE: break
    if val is not None: m.set_weights(best[2])
    return m, best[1], best[0]

parts = [pp.extract_both(r, MITDB) for r in rd.DS1]
cat = lambda k: np.concatenate([p[k] for p in parts])
X1, R1, y1, g1 = cat('X_f32').astype(np.float32), cat('R_f32').astype(np.float32), cat('y'), cat('rec')
folds = rd.make_folds({r: y1[g1 == r] for r in rd.DS1})
for seed, k in JOBS:
    t0 = time.time(); V = folds[k]; Tr = [r for r in rd.DS1 if r not in V]
    inner = rd.make_folds({r: y1[g1 == r] for r in Tr}, k=3)            # CV 3 fold BEN TRONG T_k (208 luon o train)
    eps, offs, f1s = [], [], []
    for I in inner:
        tmask = np.isin(g1, Tr) & ~np.isin(g1, I); imask = np.isin(g1, I)
        m1, ep, f1i = train(X1[tmask], R1[tmask], y1[tmask], MAX_EPOCHS, val=(X1[imask], R1[imask], y1[imask]), seed=seed)
        lg = m1.predict([X1[imask][..., None], rr_norm(R1[imask])], batch_size=4096, verbose=0)
        off, f1o = tune_bias(lg, y1[imask]); eps.append(ep); offs.append(off); f1s.append(f1i)
        tf.keras.backend.clear_session()
    ep = int(round(np.mean(eps))); off = np.mean(offs, 0)
    Tmask = np.isin(g1, Tr)
    m2, _, _ = train(X1[Tmask], R1[Tmask], y1[Tmask], ep, val=None, seed=seed)
    m2.save(os.path.join(OUT, f'F2_nfold{k}_s{seed}.keras'))
    meta = dict(seed=seed, fold=k, val_recs=V, inner_folds=inner, inner_epochs=eps, epochs=ep, inner_f1=f1s,
                inner_offsets=[o.tolist() for o in offs], offset=off.tolist(), minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(OUT, f'F2_nfold{k}_s{seed}.json'), 'w'), indent=1)
    print(json.dumps(meta), flush=True)
    tf.keras.backend.clear_session()
