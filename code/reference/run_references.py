"""Optical, transfer and counting references computed independently of any device simulator (Fig. 4, Table 3,
Tables A.2-A.3): analytic continuum reference, discrete consumers, valence-basis assertions. Writes results/reference/."""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,sys
from datetime import datetime,timezone
from math import cos, sin, tan, tanh, sqrt, pi, exp

import numpy as np
from scipy.linalg import eigh_tridiagonal

for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[variable]='4'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
# Each run writes one new directory <out-root>/<name> (default out-root: results/).
OUT_ROOT=ROOT.parents[1]/'results'


# ---------------------------------------------------------------- continuum reference (standard-library arithmetic)
def bisect(fun, left, right):
    fl, fr=fun(left),fun(right)
    if fl*fr>=0:raise ValueError('Reference root is not bracketed')
    for _ in range(100):
        mid=(left+right)/2
        fm=fun(mid)
        if fm==0 or right-left<2e-15:return mid
        if fl*fm<0:right=mid
        else:left=mid;fl=fm
    return (left+right)/2


def isolated_well(C, width, barrier):
    a=width/2
    upper=min(sqrt(barrier/C),pi/(2*a))*(1-1e-12)
    k=bisect(lambda q:q*tan(q*a)-sqrt(barrier/C-q*q),1e-12,upper)
    energy=C*k*k
    decay=sqrt((barrier-energy)/C)
    norm=1/sqrt(a+sin(2*k*a)/(2*k)+cos(k*a)**2/decay)
    return {'energy_eV':energy,'k_nm_inverse':k,'decay_nm_inverse':decay,
            'norm_nm_minus_half':norm,'tail_nm_minus_half':norm*cos(k*a)}


def transfer(C, width, barrier, separation):
    ref=isolated_well(C,width,barrier)
    value=2*C*ref['decay_nm_inverse']*ref['tail_nm_minus_half']**2*exp(-ref['decay_nm_inverse']*separation)
    return dict(ref,T_abs_eV=value,T2_eV2=value*value)


def double_well_levels(C,width,barrier,separation):
    def residual(energy,odd):
        k=sqrt(energy/C);decay=sqrt((barrier-energy)/C)
        t=tanh(decay*separation/2)
        r=decay/t if odd else decay*t
        u=cos(k*width)+(r/k)*sin(k*width)
        v=-k*sin(k*width)+r*cos(k*width)
        return v+decay*u
    roots=[]
    for odd in [False,True]:
        left=barrier*1e-10;fl=residual(left,odd)
        for i in range(1,4097):
            right=barrier*(i/4096)*(1-1e-10)
            fr=residual(right,odd)
            if fl*fr<0:
                roots.append(bisect(lambda e:residual(e,odd),left,right));break
            left=right;fl=fr
        else:raise ValueError('No bound-state root found')
    return {'even_eV':roots[0],'odd_eV':roots[1], 'half_splitting_eV':(roots[1]-roots[0])/2}


# ---------------------------------------------------------------- discrete consumers and interface guards
BASIS=('S_up','X_up','Y_up','Z_up','S_down','X_down','Y_down','Z_down')


def validate_envelopes(c,z,ids,basis=BASIS):
    """c[state, grid, component], Euclidean norm per whole state; z in nm."""
    c=np.asarray(c);z=np.asarray(z)
    if c.ndim!=3 or c.shape[1]!=len(z) or c.shape[2]!=len(basis):
        raise ValueError('Envelope dimensions do not match grid and declared basis')
    if len(ids)!=len(c) or len(set(ids))!=len(ids):raise ValueError('State IDs must be unique')
    spacing=np.diff(z)
    if len(spacing)<2 or np.any(spacing<=0) or not np.allclose(spacing,spacing[0],rtol=1e-12,atol=1e-13):
        raise ValueError('Only strictly increasing uniform nm grids are supported')
    if not np.all(np.isfinite(c)) or not np.allclose(np.sum(abs(c)**2,axis=(1,2)),1,rtol=0,atol=1e-11):
        raise ValueError('Expected finite whole-state Euclidean normalized coefficients')
    return float(spacing[0])


def optical_operator(EP,C0,polarization):
    e=np.asarray(polarization,dtype=complex)
    if e.shape!=(3,) or not np.isclose(np.vdot(e,e).real,1,rtol=0,atol=1e-12):
        raise ValueError('Polarization must be a normalized Cartesian vector')
    if EP<=0 or C0<=0:raise ValueError('Positive eV Kane parameter and eV nm2 constant required')
    D=np.zeros((8,8),complex);P=np.sqrt(C0*EP)
    # Each Cartesian operator is Hermitian. Complex polarization combines
    # both directed entries with e_j, not with conjugate(e_j).
    for spin in [0,4]:
        for axis in range(3):
            D[spin,spin+axis+1]=1j*P*e[axis]
            D[spin+axis+1,spin]=-1j*P*e[axis]
    return D


def optical_amplitude(c_cb,c_vb,z,EP,C0,polarization,convert=True):
    h=validate_envelopes(c_cb,z,[f'cb_{i}' for i in range(len(c_cb))])
    validate_envelopes(c_vb,z,[f'vb_{i}' for i in range(len(c_vb))])
    D=optical_operator(EP,C0,polarization)
    a=c_cb/np.sqrt(h) if convert else c_cb
    b=c_vb/np.sqrt(h) if convert else c_vb
    amplitude=h*np.einsum('aic,cd,bid->ab',a.conj(),D,b)
    return amplitude


def optical_matrix(c_cb,c_vb,z,EP,C0,polarization,convert=True):
    return abs(optical_amplitude(c_cb,c_vb,z,EP,C0,polarization,convert))**2


def fd_states(C,width,barrier,separation,h,half_box,which):
    n=round(2*half_box/h)
    if abs(n*h-2*half_box)>1e-10:raise ValueError('Grid must partition fixed box')
    z=-half_box+np.arange(1,n)*h
    centers={'left':[-(width+separation)/2],'right':[(width+separation)/2],
             'double':[-(width+separation)/2,(width+separation)/2]}[which]
    fraction=np.zeros(len(z))
    for center in centers:
        overlap=np.maximum(0,np.minimum(z+h/2,center+width/2)-np.maximum(z-h/2,center-width/2))
        fraction+=overlap/h
    v=barrier*(1-np.clip(fraction,0,1))
    t=C/h**2
    eigenvalues,states=eigh_tridiagonal(2*t+v,np.full(len(z)-1,-t),select='i',
                                       select_range=(0,1 if which=='double' else 0),tol=1e-13)
    return z,eigenvalues,states


def transfer_from_states(C,z,c_left,c_right,h,surface,convert=True):
    index=int(np.argmin(abs(z-surface)))
    if abs(z[index]-surface)>1e-9 or index in (0,len(z)-1):
        raise ValueError('Surface must be an interior grid node')
    l=c_left/np.sqrt(h) if convert else c_left
    r=c_right/np.sqrt(h) if convert else c_right
    dl=(l[index+1]-l[index-1])/(2*h)
    dr=(r[index+1]-r[index-1])/(2*h)
    return float(abs(C*(l[index].conjugate()*dr-dl.conjugate()*r[index])))


# ---------------------------------------------------------------- valence-basis assertions
# Absolute tolerance of the three assertions and of the operator-construction checks,
# fixed here before the first run (same size as the optical relative budget,
# data/reference/model.json checks.optical_relative_error).
TOL = 1e-12
J2_J32 = 15 / 4
J2_J12 = 3 / 4


def angular_momentum():
    """Explicit L, S, J = L + S (lists of three 8x8 matrices) and J^2 in the basis BASIS.

    Index = 4*spin + orbital with spin in (up, down) and orbital in (S, X, Y, Z),
    so every operator is kron(spin part, orbital part)."""
    eps = np.zeros((3, 3, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[a, b, c] = 1.0
        eps[a, c, b] = -1.0
    orbital = []
    for k in range(3):
        m = np.zeros((4, 4), complex)
        m[1:, 1:] = -1j * eps[k]          # (L_k)_ij = -i eps_kij on (X, Y, Z); L = 0 on S
        orbital.append(m)
    pauli = (np.array([[0, 1], [1, 0]], complex),
             np.array([[0, -1j], [1j, 0]], complex),
             np.array([[1, 0], [0, -1]], complex))
    L = [np.kron(np.eye(2), orbital[k]) for k in range(3)]
    S = [np.kron(pauli[k] / 2, np.eye(4)) for k in range(3)]
    J = [L[k] + S[k] for k in range(3)]
    J2 = sum(Jk @ Jk for Jk in J)
    return L, S, J, J2


def _commutator_residual(ops):
    """max |[A_x, A_y] - i A_z| over the three cyclic pairs."""
    worst = 0.0
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        r = ops[a] @ ops[b] - ops[b] @ ops[a] - 1j * ops[c]
        worst = max(worst, float(np.max(np.abs(r))))
    return worst


def split_off_states():
    """|1/2, +1/2> = [(X + iY)_down + Z_up]/sqrt3 and |1/2, -1/2> = [(X - iY)_up - Z_down]/sqrt3."""
    so = np.zeros((2, 8), complex)
    so[0, 5] = 1 / np.sqrt(3)
    so[0, 6] = 1j / np.sqrt(3)
    so[0, 3] = 1 / np.sqrt(3)
    so[1, 1] = 1 / np.sqrt(3)
    so[1, 2] = -1j / np.sqrt(3)
    so[1, 7] = -1 / np.sqrt(3)
    return so


def opposite_sign_lh_states():
    """Control only: light-hole vectors with the opposite relative sign,
        sqrt(2/3)Z_up + (X_down + iY_down)/sqrt6 and sqrt(2/3)Z_down - (X_up - iY_up)/sqrt6.
        Not used in any computation of the reference.
    """
    lh = np.zeros((2, 8), complex)
    lh[0, 3] = lh[1, 7] = np.sqrt(2 / 3)
    lh[0, 5] = 1 / np.sqrt(6)
    lh[0, 6] = 1j / np.sqrt(6)
    lh[1, 1] = -1 / np.sqrt(6)
    lh[1, 2] = 1j / np.sqrt(6)
    return lh


def _c(z):
    z = complex(z)
    return [z.real, z.imag]


def _vec(v):
    return [_c(x) for x in v]


def _expect(op, v):
    return complex(np.vdot(v, op @ v))


def _state_quantities(v, J, J2, so):
    j2 = _expect(J2, v)
    return {'J2': j2.real, 'J2_imag': j2.imag, 'Jz': _expect(J[2], v).real,
            'J2_eigen_residual': float(np.linalg.norm(J2 @ v - j2.real * v)),
            'norm2': float(np.vdot(v, v).real),
            'overlap_SO_up': _c(np.vdot(so[0], v)), 'overlap_SO_down': _c(np.vdot(so[1], v)),
            'split_off_weight': float(abs(np.vdot(so[0], v)) ** 2 + abs(np.vdot(so[1], v)) ** 2)}


def valence_basis_record(hh, lh):
    """JSON-safe record of the valence-basis assertions for the Bloch vectors hh (2x8) and lh (2x8)."""
    hh = np.asarray(hh, complex)
    lh = np.asarray(lh, complex)
    L, S, J, J2 = angular_momentum()
    so = split_off_states()
    construction = {
        'max_abs_commutator_residual': {'L': _commutator_residual(L), 'S': _commutator_residual(S),
                                        'J': _commutator_residual(J)},
        'max_abs_nonhermiticity_J': max(float(np.max(np.abs(Jk - Jk.conj().T))) for Jk in J),
        'J2_eigenvalues': [float(x) for x in np.linalg.eigvalsh(J2)],
        'split_off_J2': [_expect(J2, v).real for v in so],
        'split_off_gram_max_abs_deviation': float(np.max(np.abs(so.conj() @ so.T - np.eye(2)))),
    }
    construction_ok = (max(construction['max_abs_commutator_residual'].values()) <= TOL
                       and construction['max_abs_nonhermiticity_J'] <= TOL
                       and all(abs(x - J2_J12) <= TOL for x in construction['split_off_J2'])
                       and construction['split_off_gram_max_abs_deviation'] <= TOL)
    names = ('HH_up', 'HH_down', 'LH_up', 'LH_down')
    states = [hh[0], hh[1], lh[0], lh[1]]
    gram = np.array([[np.vdot(a, b) for b in states] for a in states])
    gram_dev = float(np.max(np.abs(gram - np.eye(4))))
    so_overlaps = {f'<SO_{s}|{n}>': _c(np.vdot(so[i], v))
                   for n, v in (('LH_up', lh[0]), ('LH_down', lh[1]))
                   for i, s in enumerate(('up', 'down'))}
    so_max = max(abs(complex(*x)) for x in so_overlaps.values())
    per_state = {n: _state_quantities(v, J, J2, so) for n, v in zip(names, states)}
    j2_dev = max(abs(complex(per_state[n]['J2'], per_state[n]['J2_imag']) - J2_J32) for n in names)
    assertions = {'pairwise_orthonormal': bool(gram_dev <= TOL),
                  'lh_split_off_overlap_zero': bool(so_max <= TOL),
                  'J2_equals_15_over_4': bool(j2_dev <= TOL)}
    opposite_lh = opposite_sign_lh_states()
    control = {n: _state_quantities(v, J, J2, so) for n, v in (('LH_up', opposite_lh[0]), ('LH_down', opposite_lh[1]))}
    for i, n in enumerate(('LH_up', 'LH_down')):
        control[n]['weight_on_used_state'] = float(abs(np.vdot(lh[i], opposite_lh[i])) ** 2)
    return {
        'checks': 'orthonormality, zero split-off overlap of the light holes, <J^2> = 15/4',
        'convention': ('(L_k)_ij = -i eps_kij on (X, Y, Z), L = 0 on S; S_k = sigma_k/2 (standard Pauli); '
                       'J = L + S; hbar = 1; operator = kron(spin, orbital)'),
        'basis': list(BASIS),
        'tolerance_abs': TOL,
        'operator_construction': construction,
        'operator_construction_ok': bool(construction_ok),
        'states': {n: _vec(v) for n, v in zip(names, states)},
        'gram_matrix': [[_c(x) for x in row] for row in gram],
        'gram_max_abs_deviation': gram_dev,
        'lh_split_off_overlaps': so_overlaps,
        'lh_split_off_overlap_max_abs': so_max,
        'J2_max_abs_deviation_from_15_over_4': j2_dev,
        'per_state': per_state,
        'assertions': assertions,
        'all_assertions_passed': bool(construction_ok and all(assertions.values())),
        'control_opposite_sign_lh': {
            'note': 'light-hole vectors with the opposite relative sign, recorded as a control only (expected split-off weight 8/9)',
            'states': {n: _vec(v) for n, v in (('LH_up', opposite_lh[0]), ('LH_down', opposite_lh[1]))},
            'per_state': control,
            'expected_split_off_weight': 8 / 9,
            'expected_J2': 13 / 12,
        },
    }


def assert_valence_basis(record):
    """Raise ValueError unless the operator construction and the three assertions hold."""
    failed = [k for k, v in record['assertions'].items() if not v]
    if not record['operator_construction_ok']:
        failed.insert(0, 'operator_construction')
    if failed:
        raise ValueError('Valence-basis assertion failed: ' + ', '.join(failed))
    return True


def single_branch_rows(optical_matrix, cb, bands, op, C0):
    """Single-branch optical intensities from the package's own consumer call.

        The grid, envelope and Bloch-array construction repeat the optical loop of run_references.py
        verbatim; only the per-branch bookkeeping is added. m = optical_matrix(...) has rows = conduction
        states (S_up, S_down) and columns = valence branches; each row of the output is one valence
        branch summed over both conduction-spin final states, and its ratio to P^2 = C0*EP.
    """
    rows = []
    P2 = C0 * op['kane_energy_eV']
    for h in op['grid_nm']:
        L = op['well_width_nm']
        z = np.arange(1, round(L / h)) * h
        a = np.sqrt(2 * h / L) * np.sin(np.pi * z / L)
        cca = np.einsum('i,ac->aic', a, cb)
        for band, bloch in bands:
            cvb = np.einsum('i,ac->aic', a, bloch)
            for pol, e in [('TE', (0, 1, 0)), ('TM', (0, 0, 1))]:
                m = optical_matrix(cca, cvb, z, op['kane_energy_eV'], C0, e)
                for branch in range(m.shape[1]):
                    s = float(m[0, branch] + m[1, branch])
                    rows.append({'h_nm': h, 'band': band, 'polarization': pol, 'valence_branch': branch + 1,
                                 'M2_to_S_up_eV2_nm2': float(m[0, branch]),
                                 'M2_to_S_down_eV2_nm2': float(m[1, branch]),
                                 'M2_branch_eV2_nm2': s, 'branch_over_P2': s / P2})
    return rows


# ---------------------------------------------------------------- program
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',default=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%f'))
    ap.add_argument('--out-root',default=str(OUT_ROOT),help='Parent of the new output directory')
    args=ap.parse_args()
    if Path(args.name).name!=args.name or args.name in ('.','..'):raise ValueError('One directory name required')
    out=Path(args.out_root)/args.name;out.mkdir(parents=True,exist_ok=False)
    import scipy
    from scipy.integrate import quad

    config_path=ROOT.parents[1]/'data'/'reference'/'model.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8'));checks=cfg['checks']
    dump(out/'execution_inputs.json',{'criteria':cfg,'sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}})
    dump(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
         'platform':platform.platform(),
         'threads':{v:os.environ[v] for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}})
    C0=cfg['hbar2_over_2m0_eV_nm2'];op=cfg['optical'];dw=cfg['double_well']
    C=C0/dw['mass_m0'];w=dw['well_width_nm'];V=dw['barrier_eV'];b=dw['barrier_width_nm']
    ref=transfer(C,w,V,b);exact=double_well_levels(C,w,V,b)
    transfer_rows=[]
    for h in dw['grid_nm']:
        z,el,cl=fd_states(C,w,V,b,h,dw['box_half_width_nm'],'left')
        zr,er,cr=fd_states(C,w,V,b,h,dw['box_half_width_nm'],'right')
        zd,ed,cd=fd_states(C,w,V,b,h,dw['box_half_width_nm'],'double')
        values=[transfer_from_states(C,z,cl[:,0],cr[:,0],h,s) for s in dw['surfaces_nm']]
        correct=values[1];bad=transfer_from_states(C,z,cl[:,0],cr[:,0],h,0,False)
        transfer_rows.append({'h_nm':h,'isolated_E_eV':float(el[0]),'T_eV':correct,
             'T2_eV2':correct**2,'T_reference_eV':ref['T_abs_eV'],
             'T_relative_error':abs(correct/ref['T_abs_eV']-1),
             'T_raw_as_continuum_eV':bad,'squared_normalization_factor_residual':abs((bad/correct)**2/h**2-1),
             'T_surface_minus2_eV':values[0],'T_surface_plus2_eV':values[2],
             'surface_relative_spread':(max(values)-min(values))/correct,
             'double_even_eV':float(ed[0]),'double_odd_eV':float(ed[1]),
             'double_energy_relative_error':max(abs(ed[0]/exact['even_eV']-1),abs(ed[1]/exact['odd_eV']-1))})
    h=dw['grid_nm'][-1]
    z,_,cl=fd_states(C,w,V,b,h,36,'left');_,_,cr=fd_states(C,w,V,b,h,36,'right')
    extended=transfer_from_states(C,z,cl[:,0],cr[:,0],h,0)
    box_error=abs(extended/transfer_rows[-1]['T_eV']-1)
    weak_rows=[]
    for barrier_width in dw['barrier_crosscheck_nm']:
        tr=transfer(C,w,V,barrier_width);de=double_well_levels(C,w,V,barrier_width)
        weak_rows.append({'barrier_nm':barrier_width,'T_eV':tr['T_abs_eV'],**de,
                         'half_splitting_vs_T_relative_difference':abs(de['half_splitting_eV']/tr['T_abs_eV']-1)})
    csv_write(out/'transfer_grids.csv',transfer_rows);csv_write(out/'weak_coupling.csv',weak_rows)
    dump(out/'continuum_reference.json',{'isolated_and_transfer':ref,'full_double_well':exact,
                                        'extended_box_T_eV':extended,'box_relative_difference':box_error})
    # Explicit Bloch vectors; the independent expected intensities use coefficients only.
    cb=np.eye(8,dtype=complex)[[0,4]]
    hh=np.zeros((2,8),complex);hh[0,1]=hh[1,5]=1/np.sqrt(2);hh[0,2]=1j/np.sqrt(2);hh[1,6]=-1j/np.sqrt(2)
    # Standard Condon-Shortley light holes sqrt(2/3)Z_up-(X_down+iY_down)/sqrt6 and
    # sqrt(2/3)Z_down+(X_up-iY_up)/sqrt6 (the opposite relative sign would give 1/9 LH + 8/9 SO).
    lh=np.zeros((2,8),complex);lh[0,3]=lh[1,7]=np.sqrt(2/3)
    lh[0,5]=-1/np.sqrt(6);lh[0,6]=-1j/np.sqrt(6);lh[1,1]=1/np.sqrt(6);lh[1,2]=-1j/np.sqrt(6)
    # Orthonormality, zero split-off overlap and <J^2>=15/4 asserted before any optical use.
    basis_record=valence_basis_record(hh,lh);dump(out/'valence_basis_checks.json',basis_record);assert_valence_basis(basis_record)
    optical_rows=[];zero_rows=[]
    expected={'HH':{'TE':1.,'TM':0.},'LH':{'TE':1/3,'TM':4/3}}
    for h in op['grid_nm']:
        L=op['well_width_nm'];z=np.arange(1,round(L/h))*h
        a=np.sqrt(2*h/L)*np.sin(np.pi*z/L);other=np.sqrt(2*h/L)*np.sin(2*np.pi*z/L)
        cca=np.einsum('i,ac->aic',a,cb)
        for band,bloch in [('HH',hh),('LH',lh)]:
            cvb=np.einsum('i,ac->aic',a,bloch)
            for pol,e in [('TE',(0,1,0)),('TM',(0,0,1))]:
                q=float(optical_matrix(cca,cvb,z,op['kane_energy_eV'],C0,e).sum())
                bad=float(optical_matrix(cca,cvb,z,op['kane_energy_eV'],C0,e,False).sum())
                target=expected[band][pol]*C0*op['kane_energy_eV']
                error=abs(q/target-1) if target else abs(q)
                optical_rows.append({'h_nm':h,'band':band,'polarization':pol,'branch_count':2,
                    'M2_sum_eV2_nm2':q,'reference_eV2_nm2':target,'error':error,
                    'error_kind':'relative' if target else 'absolute',
                    'raw_as_continuum_eV2_nm2':bad,
                    'factor_residual':abs(bad/(q*h*h)-1) if q else 0})
        cvb=np.einsum('i,ac->aic',other,hh)
        q=float(optical_matrix(cca,cvb,z,op['kane_energy_eV'],C0,(0,1,0)).sum())
        zero_rows.append({'h_nm':h,'coherent_overlap':float(np.vdot(a,other).real),
                         'density_overlap':float(np.sum(abs(a)*abs(other))),
                         'HH_TE_zero_reference_M2':q,'decision':'zero reference: absolute test, no logarithmic slope'})
    csv_write(out/'optical_reference.csv',optical_rows);csv_write(out/'zero_reference.csv',zero_rows)
    # Single-branch optical intensities (same consumer call as above).
    csv_write(out/'optical_branches.csv',single_branch_rows(optical_matrix,cb,[('HH',hh),('LH',lh)],op,C0))
    # Interface negative controls exercise mistakes that norm/slope plots alone may miss.
    guard_rows=[]
    for name,action in [
        ('continuous_envelope_labeled_discrete',lambda:validate_envelopes(cca/np.sqrt(h),z,['c0','c1'])),
        ('duplicate_state_ids',lambda:validate_envelopes(cca,z,['c0','c0'])),
        ('nonuniform_grid',lambda:validate_envelopes(cca,z+np.linspace(0,.001,len(z))**2,['c0','c1'])),
        ('unnormalized_polarization',lambda:optical_matrix(cca,cvb,z,20,C0,(0,2,0)))]:
        try:action();guard_rows.append({'case':name,'rejected':False})
        except ValueError as exc:guard_rows.append({'case':name,'rejected':True,'reason':str(exc)})
    dump(out/'interface_negative_controls.json',guard_rows)
    # Single-branch 2D density, with independent radial quadrature.
    kT=8.617333262145e-5*300;alpha=C0/.055;E0=.1;mu=.12
    analytic=kT/(4*np.pi*alpha)*np.logaddexp(0,(mu-E0)/kT)
    kmax=np.sqrt((mu-E0+45*kT)/alpha)
    numerical,error=quad(lambda k:k/(1+np.exp((E0+alpha*k*k-mu)/kT))/(2*np.pi),0,kmax,epsabs=1e-13,epsrel=1e-12)
    occupation=[]
    for f in [.05,.2,.8,.95]:
        muf=E0+kT*np.log(f/(1-f));recovered=1/(1+np.exp((E0-muf)/kT))
        occupation.append({'electron_f0':f,'hole_f0':1-f,'mu_eV':muf,'recovered_f0':float(recovered)})
    dump(out/'counting_and_occupation.json',{'T_K':300,'alpha_eV_nm2':alpha,'E0_eV':E0,'mu_eV':mu,
        'single_branch_analytic_nm_minus2':float(analytic),'single_branch_quadrature_nm_minus2':numerical,
        'two_identical_branches_nm_minus2':float(2*analytic),'relative_error':float(abs(numerical/analytic-1)),
        'quadrature_error_estimate':error,'occupation_roundtrips':occupation})
    results={
        'optical_absolute_reference':all(r['error'] <= (checks['optical_relative_error'] if r['error_kind']=='relative' else checks['zero_optical_absolute_eV2_nm2']) for r in optical_rows),
        'optical_h2_factor':all(r['factor_residual']<=checks['normalization_factor_residual'] for r in optical_rows),
        'zero_coherent_reference':all(r['HH_TE_zero_reference_M2']<=checks['zero_optical_absolute_eV2_nm2'] for r in zero_rows),
        'finest_transfer_reference':transfer_rows[-1]['T_relative_error']<=checks['finest_transfer_amplitude_relative_error'],
        'finest_full_spectrum_reference':transfer_rows[-1]['double_energy_relative_error']<=checks['finest_double_well_energy_relative_error'],
        'transfer_h2_factor':all(r['squared_normalization_factor_residual']<=checks['normalization_factor_residual'] for r in transfer_rows),
        'surface_spread':all(r['surface_relative_spread']<=checks['surface_relative_spread'] for r in transfer_rows),
        'box_extension':box_error<=checks['box_extension_relative_difference'],
        'weak_limit_crosscheck':weak_rows[-1]['half_splitting_vs_T_relative_difference']<=checks['weak_barrier_12nm_half_splitting_relative_difference'],
        'interface_guards':all(r['rejected'] for r in guard_rows),
        'counting_reference':abs(numerical/analytic-1)<1e-10,
        'occupation_roundtrip':all(abs(r['recovered_f0']-r['electron_f0'])<1e-12 for r in occupation)}
    results={name:bool(value) for name,value in results.items()}
    dump(out/'summary.json',{'passed':all(results.values()),'checks':results,
         'scope':'Declared optical model and scalar transfer amplitude; no many-band device-rate validation',
         'preexisting_scientific_failures':'none altered; every check is reported as computed'})
    print(json.dumps({'output':str(out),'passed':all(results.values()),'checks':results,
                      'T_reference_meV':ref['T_abs_eV']*1000,
                      'finest_relative_error':transfer_rows[-1]['T_relative_error']}))
    return 0 if all(results.values()) else 1


if __name__=='__main__':raise SystemExit(main())
