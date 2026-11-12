import zipfile,h5py,io,json,numpy as np, pa_adapt as pa
FE={1:3,2:3,3:4,4:3,5:3}; OFF=np.array([0.0,0.06666666666666687,-0.5333333333333332,-1.0999999999999999])
D=np.load('/tmp/claude-0/-home-claude/1dc7cf9a-f8f8-5a53-afa8-ed0c0ae36b51/scratchpad/F2A_BOARD_RUN/f2a_board_data.npz')
y,rec,t=D['y'],D['rec'],D['t']; cal=pa.split_first(t,rec,5); ev=~cal
ANC_Y=np.repeat(np.arange(4),25)
HP=json.load(open('hp.json'))
def head(seed):
    z=zipfile.ZipFile(f'F2_rr8_str_s{seed}.keras'); f=h5py.File(io.BytesIO(z.read('model.weights.h5')),'r')
    W=f['layers/dense_2/vars/0'][()]; b=f['layers/dense_2/vars/1'][()]
    wo=np.clip(np.round(W*128),-128,127); b8=np.clip(np.round(b*128),-128,127)
    bq=np.round((b8/128+OFF)*2**14)
    return wo/128.0, bq/2**14
def seed_data(seed):
    z=np.load(f'exp4_emb_int_s{seed}.npz'); s=2.0**FE[seed]
    W0,b0=head(seed); return z['E']/s, W0, b0, (z['A']/s, ANC_Y)
