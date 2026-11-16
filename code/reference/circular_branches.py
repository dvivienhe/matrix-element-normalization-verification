"""Circular-polarization branch check of the optical reference (Fig. 3): coherent squared amplitude of the two
heavy-hole branches and the incoherent sum as a negative control. Writes results/circular_branches/."""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, sys
from datetime import datetime, timezone

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT.parents[1]/'results'
CFG = {'grid_nm': [0.4, 0.2, 0.1, 0.05], 'optical_absolute_eV2_nm2': 1e-24,
       'optical_relative': 1e-12, 'component_residual': 1e-10}


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csvout(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def optical(out, cfg):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from run_references import optical_operator, optical_amplitude
    P2 = .0380998211615486*20
    cb = np.eye(8, dtype=complex)[[0,4]]
    hh = np.zeros((2,8), complex)
    hh[0,1] = hh[1,5] = 1/np.sqrt(2)
    hh[0,2], hh[1,6] = 1j/np.sqrt(2), -1j/np.sqrt(2)
    e = np.array([1,1j,0])/np.sqrt(2)
    D = optical_operator(20,.0380998211615486,e)
    conjugacy = float(np.max(abs(D.conj().T-optical_operator(20,.0380998211615486,e.conj()))))
    rows=[]
    for h in cfg['grid_nm']:
        z = np.arange(1,round(10/h))*h
        a = np.sqrt(2*h/10)*np.sin(np.pi*z/10)
        cc = np.einsum('i,ac->aic',a,cb); vv = np.einsum('i,ac->aic',a,hh)
        amp = optical_amplitude(cc,vv,z,20,.0380998211615486,e)
        # Deliberate negative control: integrate each Bloch channel coherently
        # in space, but discard cross-channel interference before summing.
        channel = np.einsum('aic,cd,bid->abcd',cc.conj(),D,vv)
        wrong = np.sum(abs(channel)**2, axis=(2,3))
        for branch in range(2):
            target = (0.,P2)[branch]
            value = float(abs(amp[branch,branch])**2)
            rows.append({'h_nm':h,'branch':branch+1,'real_M':float(amp[branch,branch].real),
                'imag_M':float(amp[branch,branch].imag),'coherent_M2':value,
                'incoherent_M2':float(wrong[branch,branch]),'expected_M2':target,
                'passed':bool(abs(value-target) <= (cfg['optical_absolute_eV2_nm2'] if target==0
                           else cfg['optical_relative']*target))})
    csvout(out/'circular_branches.csv', rows)
    result={'all_branch_checks':all(r['passed'] for r in rows),
            'adjoint_identity_passed':conjugacy<=cfg['component_residual'],
            'conjugate_polarization_adjoint_residual':conjugacy,
            'D_nonhermiticity_max':float(np.max(abs(D-D.conj().T))),
            'wrong_total_equals_right_total':bool(all(abs(rows[i]['incoherent_M2']+rows[i+1]['incoherent_M2']-P2)<1e-12*P2 for i in range(0,len(rows),2))),
            'wrong_each_branch_detected':all(abs(r['incoherent_M2']-r['expected_M2'])>.49*P2 for r in rows)}
    dump(out/'circular_checks.json',result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default=datetime.now(timezone.utc).strftime('circular_%Y%m%dT%H%M%S_%f'))
    ap.add_argument('--out-root', default=str(OUT_ROOT), help='Parent of the new output directory')
    args = ap.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('One directory name required')
    out = Path(args.out_root)/args.name
    out.mkdir(parents=True, exist_ok=False)
    import numpy, scipy
    dump(out/'circular_inputs.json', {
        'settings': CFG,
        'settings_source': 'CFG of this script: the four settings read by optical()',
        'source_sha256': {p.name: sha(p) for p in (ROOT/'run_references.py', Path(__file__).resolve())},
        'python': sys.version, 'numpy': numpy.__version__, 'scipy': scipy.__version__,
        'platform': platform.platform(),
        'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    result = optical(out, CFG)
    gates = {k: v for k, v in result.items() if isinstance(v, bool)}
    print(json.dumps({'output': str(out), 'gates': gates}))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
