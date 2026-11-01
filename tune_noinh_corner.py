import sys, json; sys.path.insert(0,'.')
from bench import Bench, ROOT
from data import natural
from metrics import score
P=ROOT/'params.json'; p=json.loads(P.read_text())
train,val,_=natural(); cache=ROOT/'runs'/'fit_natural.pkl'
log=p['tune_log']+['--- stage 5: close the no_inhibition low corner ---']
for tag,(depth,dt,gam,q) in {'sihti':(p['depth'],0.25,p['csc_gamma'],p),'heat':(p['heat']['depth'],p['heat']['dt'],p['heat']['csc_gamma'],p['heat'])}.items():
    b=Bench(train,depth,p['sigma'],cache=cache,dt=dt); Y=b.observe_set(val,seed=11)
    r={(q['noinh_lam'],q['noinh_scale']):None}
    best=None
    for lam in (0.0003,0.001,0.003):
        for s in (0.0015,0.005,0.015):
            m=score([b.csc.solve_no_inhibition(y,lam,gam,s) for y in Y],val)['mse']
            log.append(f'[{tag}] no_inhibition lam={lam:g} scale={s:g}: val mse {m:.6f}')
            if best is None or m<best[0]: best=(m,lam,s)
    q['noinh_lam'],q['noinh_scale']=best[1],best[2]
    log.append(f'[{tag}] no_inhibition final lam={best[1]:g} scale={best[2]:g} val mse {best[0]:.6f}')
p['tune_log']=log; P.write_text(json.dumps(p,indent=2))
print('\n'.join(log[-20:]))
