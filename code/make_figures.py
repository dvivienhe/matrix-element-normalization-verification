"""Draw the eight figures of the article (Figs. 1-6, S1, S2) at final size from the computed results and the
shipped data. Writes results/figures/Figure_*.{pdf,eps,png}."""
from pathlib import Path
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import (FixedFormatter, FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator,
                               NullFormatter, NullLocator)
from matplotlib.transforms import ScaledTranslation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / 'results'
DATA = REPO / 'data'
OUT = RESULTS / 'figures'
MM = 1 / 25.4
SOURCE_DATE_EPOCH = '0'                  # fixes the EPS %%CreationDate (PDF CreationDate is removed)


def pick_font():
    """Arial, else Liberation Sans, else Helvetica - never a silent fallback to DejaVu Sans."""
    for family in ('Arial', 'Liberation Sans', 'Helvetica'):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        if font_manager.get_font(path).family_name == family:
            return family, path
    raise SystemExit('None of Arial, Liberation Sans or Helvetica is installed.')


FONT, FONT_PATH = pick_font()
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [FONT], 'font.size': 7.5,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.sf': FONT, 'mathtext.it': f'{FONT}:italic',
    'mathtext.bf': f'{FONT}:bold', 'mathtext.fallback': 'stixsans', 'axes.unicode_minus': True,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 2.5,
    'axes.titlepad': 5.5, 'lines.linewidth': 1.0, 'lines.markersize': 4.0, 'lines.markeredgewidth': 0.8,
    'patch.linewidth': 0.6, 'hatch.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.handletextpad': 0.5, 'legend.columnspacing': 1.4,
    'legend.labelspacing': 0.35, 'legend.borderaxespad': 0.4, 'legend.borderpad': 0.3,
    'figure.dpi': 200, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'path.simplify': False,
})
FS = {'tick': 8.6, 'label': 9.0, 'legend': 8.6, 'note': 8.6, 'letter': 9.0}   # sub-/superscripts (0.7x) stay >= 6 pt
FS_WIDE = {'tick': 8.6, 'label': 9.0, 'legend': 8.6, 'note': 8.6, 'letter': 9.0}   # 190 mm: ~0.87x in a 166-mm Word column
OI = {'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
      'vermillion': '#D55E00', 'purple': '#CC79A7', 'black': '#000000'}                 # Okabe-Ito
CORRECT = dict(color=OI['blue'], marker='o', ls='-')
OMITTED = dict(color=OI['vermillion'], marker='s', ls='--')
CONSTANT = dict(color=OI['purple'], marker='^', ls='-.')
NEUTRAL = dict(color='black', marker='o', ls='-')
REFERENCE = dict(color='black', ls=':', lw=0.8, zorder=1.5)       # above bars (1), below data lines (2)
GUIDE = dict(color='0.55', ls=(0, (4, 2)), lw=0.8)
DZ_LABEL = r'Grid spacing $\Delta z$ (nm)'
DZ4 = ((0.05, 0.1, 0.2, 0.4), ('0.05', '0.10', '0.20', '0.40'))
DZ_SOLVER = ((0.0625, 0.125, 0.25, 0.5), ('0.0625', '0.125', '0.25', '0.5'))


# ------------------------------------------------------------------ helpers
def read_csv(p: Path):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def new_figure(width_mm, height_mm, ncols=1, width_ratios=None):
    fig, axs = plt.subplots(1, ncols, figsize=(width_mm * MM, height_mm * MM), layout='constrained',
                            width_ratios=width_ratios)
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.05, hspace=0.02)
    return fig, axs


def finish(ax, fs, letter=None, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs['label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs['label'])
    ax.tick_params(which='both', labelsize=fs['tick'])
    if letter:                                          # moved to the panel's left label edge by align_panel_letters
        ax.set_title(letter, loc='left', fontsize=fs['letter']).set_gid('panel-letter')


def align_panel_letters(fig):
    """Put each panel letter above the left edge of its panel (y-label column), then freeze the layout."""
    letters = [t for t in fig.findobj(Text) if t.get_gid() == 'panel-letter']
    if not letters:
        return
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for t in letters:
        ax = t.axes
        t.set_x((ax.get_tightbbox(rnd).x0 - ax.bbox.x0) / ax.bbox.width)
    fig.set_layout_engine('none')


def log_x(ax, ticks):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(list(ticks[0])))
    ax.xaxis.set_major_formatter(FixedFormatter(list(ticks[1])))
    ax.xaxis.set_minor_locator(NullLocator())


def log_y(ax, lo, hi, step=1, decimal=False):
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ks = list(range(int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9)) + 1))
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in ks if (ks[-1] - k) % step == 0]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=99) if step == 1 else
                               FixedLocator([10.0 ** k for k in ks if (ks[-1] - k) % step]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}') if decimal else LogFormatterSciNotation())
    ax.yaxis.set_minor_formatter(NullFormatter())


def note(ax, x, y, s, fs, **kw):
    """In-axes annotation."""
    t = ax.text(x, y, s, fontsize=fs['note'], transform=kw.pop('transform', ax.transAxes), **kw)
    t.set_gid('note')
    return t


def sci_tex(v, digits=2):
    m, e = f'{v:.{digits}e}'.split('e')
    return rf'{m}\times10^{{{int(e)}}}'


# ------------------------------------------------------------------ Figure 1
def figure_1():
    src = RESULTS / 'controlled/diagnostic_matrix.json'
    d = json.loads(src.read_text(encoding='utf-8'))
    fig, axs = new_figure(140, 64, ncols=2)
    for ax, leg, letter in zip(axs, ('A', 'B'), ('(a)', '(b)')):
        b = d['legs'][leg]
        for key, label, style in (('C1', 'Correct conversion', CORRECT), ('C2', 'Omitted conversion', OMITTED),
                                  ('C5', r'Constant unit error ($\times10^{-2}$)', CONSTANT)):
            ax.loglog(b['dz'], np.asarray(b['rows'][key]['values']) / b['reference'][key], label=label, **style)
        ax.axhline(1, label='Reference', **REFERENCE)
        log_x(ax, DZ4)
        log_y(ax, 1e-3, 2, decimal=True)
        finish(ax, FS, letter, DZ_LABEL, r'Squared dipole $Q/Q_\mathrm{ref}$')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=FS['legend'])
    return fig


# ------------------------------------------------------------------ Figure 2
def figure_2():
    src = RESULTS / 'controlled/diagnostic_matrix.json'
    v = json.loads(src.read_text(encoding='utf-8'))['kernel_variant']
    h = np.array(v['dz'])
    slope, resid = v['slope_obs'], v['identity_worst_rel_dev']
    ratio = np.array(v['C7']) / (np.array(v['C7c']) * h ** 4)
    fig, axs = new_figure(140, 62, ncols=2)
    ax = axs[0]
    ax.loglog(h, v['C7c'], label=r'Correct conversion, $Q_\mathrm{good}$', **CORRECT)
    ax.loglog(h, v['C7'], label=r'Omitted conversion, $Q_\mathrm{bad}$', **OMITTED)
    log_x(ax, DZ4)
    log_y(ax, 1e-9, 1e-1, step=2)
    ax.legend(loc='lower right', fontsize=FS['legend'])
    note(ax, 0.048, 2.5e-3, f'Exponent of $Q_\\mathrm{{bad}}$: {slope:.4f}\nTolerance: 4 $\\pm$ 0.05', FS,
         ha='left', va='top', transform=ax.transData)
    finish(ax, FS, '(a)', DZ_LABEL, r'Raw squared quantity $|V_4|^2$')
    ax = axs[1]
    ax.plot(h, ratio, **NEUTRAL)
    log_x(ax, DZ4)
    ax.set(ylim=(.999, 1.001), yticks=[.999, 1, 1.001])
    ax.ticklabel_format(axis='y', useOffset=False)
    note(ax, 0.04, 0.05, f'Ratio residual: ${sci_tex(resid)}$\n' + r'$h=\Delta z/(1\,\mathrm{nm})$', FS,
         ha='left', va='bottom')
    finish(ax, FS, '(b)', DZ_LABEL, r'$Q_\mathrm{bad}/(Q_\mathrm{good}\,h^4)$')
    return fig


# ------------------------------------------------------------------ Figure 3
def figure_3():
    src = RESULTS / 'circular_branches/circular_branches.csv'
    rows = read_csv(src)
    P2 = max(float(r['expected_M2']) for r in rows)
    hs = sorted({float(r['h_nm']) for r in rows})               # ascending, like the Delta-z axes of Figs. 1, 2, 4
    fig, axs = new_figure(140, 62, ncols=2)
    x = np.arange(len(hs))
    for ax, br, letter in zip(axs, ('1', '2'), ('(a)', '(b)')):
        sel = [r for r in rows if r['branch'] == br]
        coh = [float(next(r for r in sel if float(r['h_nm']) == h)['coherent_M2']) / P2 for h in hs]
        inc = [float(next(r for r in sel if float(r['h_nm']) == h)['incoherent_M2']) / P2 for h in hs]
        exp = float(sel[0]['expected_M2']) / P2
        ax.bar(x - .19, coh, .36, color=OI['blue'], ec=OI['blue'], lw=0.6, label='Coherent summation (correct)')
        ax.bar(x + .19, inc, .36, color='white', ec=OI['vermillion'], hatch='////', lw=0.8,
               label='Incoherent summation (control)')
        ax.axhline(exp, label='Analytical value', **REFERENCE)
        ax.set_xticks(x, [f'{h:.2f}' for h in hs])
        ax.set(ylim=(0, 1.15), yticks=[0, .25, .5, .75, 1])
        finish(ax, FS, letter, DZ_LABEL, f'HH branch {br}, ' + r'$|M|^2/P^2$')
    note(axs[0], 0.03, 0.80, 'Coherent: 0 on all four grids', FS, ha='left', va='center')
    handles, labels = axs[0].get_legend_handles_labels()
    order = [labels.index(s) for s in ('Coherent summation (correct)', 'Incoherent summation (control)',
                                       'Analytical value')]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='outside upper center', ncol=3,
               fontsize=FS['legend'])
    return fig


# ------------------------------------------------------------------ Figure 4
def figure_4():
    tr = RESULTS / 'reference/transfer_grids.csv'
    weak = RESULTS / 'reference/weak_coupling.csv'
    rows, w = read_csv(tr), read_csv(weak)
    h = np.array([float(r['h_nm']) for r in rows])
    ref = float(rows[0]['T_reference_eV'])
    fig, axs = new_figure(140, 62, ncols=2)
    ax = axs[0]
    ax.loglog(h, [float(r['T2_eV2']) / ref ** 2 for r in rows], label='Correct conversion', **CORRECT)
    ax.loglog(h, [float(r['T_raw_as_continuum_eV']) ** 2 / ref ** 2 for r in rows], label='Omitted conversion',
              **OMITTED)
    ax.axhline(1, label='Independent reference', **REFERENCE)
    log_x(ax, DZ4)
    log_y(ax, 2e-4, 2, decimal=True)                             # extra decade fraction keeps the legend off the data
    ax.legend(loc='lower right', fontsize=FS['legend'])
    finish(ax, FS, '(a)', DZ_LABEL, r'$|T|^2/|T_\mathrm{ref}|^2$')
    ax = axs[1]
    ax.plot([float(r['barrier_nm']) for r in w], [float(r['half_splitting_vs_T_relative_difference']) for r in w],
            **NEUTRAL)
    log_y(ax, 1e-8, 1e-2, step=2)
    ax.set(xticks=[2, 4, 8, 12], xlim=(1, 13))
    finish(ax, FS, '(b)', r'Barrier width $b$ (nm)', r'$|\Delta E/2-|T||/|T|$')
    return fig


# ------------------------------------------------------------------ Figure 5
def figure_5():
    st, sw = RESULTS / 'case144/case144_summary_table.csv', RESULTS / 'case144/case144_sweep_table.csv'
    srows = [r for r in read_csv(st) if r['point'] == 'baseline' and r['group'] == 'primary']
    conds = ['full', '-N', '-W', '-O']
    get = lambda rows_, c, m, k: int(next(r for r in rows_ if r['condition'] == c and r['method'] == m)[k])
    wrows = [r for r in read_csv(sw) if r['group'] == 'primary' and r['point'].startswith('A:')]
    taus = sorted({float(r['point'].split('=')[1]) for r in wrows})
    at = lambda t: [r for r in wrows if float(r['point'].split('=')[1]) == t]
    fs = FS_WIDE
    fig, axs = new_figure(190, 80, ncols=2, width_ratios=[1.3, 1])
    ax = axs[0]
    x, wdt = np.arange(len(conds)), .26
    colour = {'OR': OI['orange'], 'CW': OI['blue'], 'OW': OI['green']}
    for k, m in enumerate(('OR', 'CW', 'OW')):
        full = [get(srows, c, m, 'FULL') for c in conds]
        ax.bar(x + (k - 1) * wdt, full, wdt, color=colour[m], ec='black', lw=0.5, label=f'{m} complete localization')
        if m == 'OW':
            cond = [get(srows, c, m, 'COND_CORRECT') for c in conds]
            ax.bar(x + (k - 1) * wdt, cond, wdt, bottom=full, color='white', ec=OI['green'], hatch='////', lw=0.8,
                   label='OW conditional attribution')
    ax.axhline(84, **REFERENCE)
    note(ax, x[-1] + 1.5 * wdt, 83, '84 injected errors', fs, ha='right', va='top', transform=ax.transData)
    ax.set_xticks(x, ['All checks\navailable', 'Norm check\nmissing', 'Quadrature-\nweight check\nmissing',
                      'Operator\ncheck\nmissing'])
    ax.set(ylim=(0, 92), yticks=[0, 20, 40, 60, 80])
    finish(ax, fs, '(a)', None, 'Number of cases')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper left', ncol=4, fontsize=fs['legend'])
    ax = axs[1]
    marks = {'-N': ('o', 5.6), '-W': ('s', 4.4), '-O': ('^', 3.8)}   # nested sizes: coincident points stay visible
    xs = np.array(taus) * 100
    for c in ('-N', '-W', '-O'):
        cw = [get(at(t), c, 'CW', 'FULL') for t in taus]
        ow = [get(at(t), c, 'OW', 'COND_CORRECT') for t in taus]
        orr = [get(at(t), c, 'OR', 'FULL') for t in taus]
        mk, ms = marks[c]
        ax.plot(xs, cw, color=OI['blue'], ls='-', marker=mk, ms=ms, mfc=OI['blue'], mec='white', mew=0.6)
        ax.plot(xs, ow, color=OI['green'], ls='--', marker=mk, ms=ms, mfc='white', mec=OI['green'], mew=0.8)
    ax.axvline(2, **REFERENCE)
    note(ax, 2.15, 77, 'baseline', fs, ha='left', va='center', transform=ax.transData)
    log_x(ax, ((0.1, 0.5, 1, 2, 5, 10), ('0.1', '0.5', '1', '2', '5', '10')))
    ax.set(ylim=(0, 84), yticks=[0, 20, 40, 60, 80])                  # top of the axis = the 84 injected errors
    handles = [Line2D([], [], color=OI['blue'], ls='-', label='CW complete\nlocalization'),
               Line2D([], [], color=OI['green'], ls='--', label='OW conditional\nattribution')]
    handles += [Line2D([], [], color='black', ls='none', marker=marks[c][0], ms=marks[c][1], mfc='white', mew=0.8,
                       label=lab) for c, lab in (('-N', 'Norm check\nmissing'),
                                                 ('-W', 'Quadrature-weight\ncheck missing'),
                                                 ('-O', 'Operator check\nmissing'))]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=fs['legend'],
              handlelength=2.4, labelspacing=0.8)
    finish(ax, fs, '(b)', r'Reference budget $\tau_g$ (%)', 'Number of cases')
    return fig


# ------------------------------------------------------------------ Figure 6
def figure_6():
    n5, ident = RESULTS / 'solcore_interface/solcore_results.json', DATA / 'solcore/identity.json'
    r5 = json.loads(n5.read_text(encoding='utf-8'))
    ladder = json.loads(ident.read_text(encoding='utf-8'))['ladder_nm']
    a_ref = r5['reference']['dipole_amplitude_nm']
    lad = np.array([0.5, 0.25, 0.125, 0.0625])                  # constant, cross-checked against the run record
    if lad.tolist() != ladder:
        raise SystemExit(f'Solcore ladder mismatch: {ladder}')
    fig, axs = new_figure(140, 64, ncols=2)
    for var, style, lab in (('node', OMITTED, 'Node-sampled potential'), ('cell', CORRECT, 'Cell-averaged potential')):
        cv = r5['convergence'][var]
        axs[0].plot(lad, cv['amplitude_nm'], label=lab, **style)
        axs[1].loglog(lad, np.abs(cv['relative_deviation_vs_reference']), label=lab, **style)
    axs[0].axhline(a_ref, label=f'Analytical reference, {a_ref:.6f} nm', **REFERENCE)
    log_x(axs[0], DZ_SOLVER)
    axs[0].ticklabel_format(axis='y', useOffset=False)
    finish(axs[0], FS, '(a)', DZ_LABEL, 'Dipole amplitude (nm)')
    g = np.array([0.0625, 0.5])                                  # guides to the eye
    axs[1].loglog(g, 0.0047 * g / 0.0625, zorder=1, **GUIDE)
    axs[1].loglog(g, 5.44e-5 * (g / 0.0625) ** 2, zorder=1, **GUIDE)
    note(axs[1], 0.17, 0.017, 'first order', FS, ha='right', va='bottom', transform=axs[1].transData)
    note(axs[1], 0.21, 3.4e-4, 'second order', FS, ha='left', va='top', transform=axs[1].transData)
    log_x(axs[1], DZ_SOLVER)
    log_y(axs[1], 2e-5, 0.1)
    finish(axs[1], FS, '(b)', DZ_LABEL, '|Relative amplitude deviation|')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=FS['legend'])
    return fig


# ------------------------------------------------------------------ Figure S1
def text_colour_on(hex_colour):
    lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(hex_colour[i:i + 2], 16) / 255) for i in (1, 3, 5))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if 1.05 / (y + 0.05) > (y + 0.05) / 0.05 else 'black'


def figure_s1():
    src = RESULTS / 'mixed_error_set/mixed_error_set.json'
    rows = json.loads(src.read_text(encoding='utf-8'))['rows']
    n = len(rows)
    methods = ('B1', 'B1n', 'B2', 'B3', 'P', 'Pp')
    columns = ('Interface\nrecords\nonly', 'Numerical\ninterface\naudit', 'Exponent\nfit only',
               'Reference\ncomparison\nonly', 'Combined\nworkflow\n(P)', 'Combined +\nasymptotic\nguards (P′)')
    cat = {'CORRECT-NEG': ('CN', OI['green'], 'correct negative'),
           'DETECTED': ('D', OI['blue'], 'detected, class correct'),
           'MISCLASSIFIED': ('MC', OI['yellow'], 'detected, class missing (°) or wrong (†)'),
           'FALSE-ALARM': ('FA', OI['orange'], 'false alarm'),
           'MISSED': ('M', OI['vermillion'], 'missed'),
           'MISSED-BY-CONSTRUCTION': ('M*', OI['purple'], 'missed by construction'),
           'UNDECIDABLE-appropriate': ('U+', OI['sky'], 'not decided (applicability guard)'),
           'UNDECIDABLE-unhelpful': ('U−', '#999999', 'not decided (unhelpful)'),
           'UNDEFINED': ('—', '#BBBBBB', 'undefined')}
    l2mark = {'CLASS-CORRECT': '', 'CLASS-NONE': '°', 'CLASS-WRONG': '†'}
    prov = {'regression-control': 'R', 'primitive-new': 'P', 'combination-new': 'C', 'guard-new': 'G'}
    fsz = 7.5
    fig = plt.figure(figsize=(140 * MM, 168 * MM), layout='constrained')
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72)
    ax = fig.add_subplot()
    for i, r in enumerate(rows):
        for j, m in enumerate(methods):
            s = r['scores'][m]
            code, colour, _ = cat[s['composite']]
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=colour, edgecolor='white', lw=1.0))
            ax.text(j + .5, n - 1 - i + .5, code + (l2mark.get(s['L2'], '') if s['L2'] else ''), ha='center',
                    va='center', fontsize=fsz, color=text_colour_on(colour))
    ax.set(xlim=(0, len(methods)), ylim=(0, n))
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(methods)) + .5, columns)
    ax.set_yticks([])
    ax.tick_params(length=0, labelsize=fsz)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # row labels split into left-aligned columns: [provenance] case  states/form  errors
    fields = [(f"[{prov[r['truth']['provenance']]}]", r['case'], f"{r['leg']}/{r['consumer']}",
               '+'.join(r['errors']) or 'none') for r in rows]
    renderer = fig.canvas.get_renderer()
    prop = font_manager.FontProperties(size=fsz)
    width_pt = lambda s: renderer.get_text_width_height_descent(s, prop, ismath=False)[0] * 72 / fig.dpi
    col_w = [max(width_pt(f[k]) for f in fields) for k in range(4)]
    offsets, xpos = [], -6.0
    for k in reversed(range(4)):
        xpos -= col_w[k]
        offsets.insert(0, xpos)
        xpos -= 5.0
    for i, f in enumerate(fields):
        for k in range(4):
            ax.text(0, n - 1 - i + .5, f[k], ha='left', va='center', fontsize=fsz,
                    transform=ax.get_yaxis_transform() + ScaledTranslation(offsets[k] / 72, 0, fig.dpi_scale_trans))
    present = [k for k in cat if any(r['scores'][m]['composite'] == k for r in rows for m in methods)]
    fig.legend(handles=[Patch(facecolor=cat[k][1], edgecolor='none', label=f'{cat[k][0]}  {cat[k][2]}')
                        for k in present], loc='outside lower center', ncol=2, fontsize=fsz, handlelength=1.6)
    return fig


# ------------------------------------------------------------------ Figure S2
def figure_s2():
    src = DATA / 'public_solver_scan.json'
    u = json.loads(src.read_text(encoding='utf-8'))
    fig, axs = new_figure(140, 60, ncols=2)
    for ax, key, field, name, letter in zip(axs, ('solcore', 'aestimo'), ('z01_nm', 'z_if_nm'),
                                            ('Solcore', 'Aestimo'), ('(a)', '(b)')):
        b = u['targets'][key]
        ax.plot(b['ladder'], [r[field] for r in b['rows']], **NEUTRAL)
        log_x(ax, DZ_SOLVER)
        ax.set_xlim(0.05, 0.6)
        ax.ticklabel_format(axis='y', useOffset=False)
        finish(ax, FS, letter, DZ_LABEL, f'{name} dipole amplitude (nm)')
    return fig


FIGURES = [  # (file stem, label, part, builder)
    ('Figure_1', 'Figure 1', 'main text', figure_1),
    ('Figure_2', 'Figure 2', 'main text', figure_2),
    ('Figure_3', 'Figure 3', 'main text', figure_3),
    ('Figure_4', 'Figure 4', 'main text', figure_4),
    ('Figure_5', 'Figure 5', 'main text', figure_5),
    ('Figure_6', 'Figure 6', 'main text', figure_6),
    ('Figure_S1', 'Figure S1', 'supplement', figure_s1),
    ('Figure_S2', 'Figure S2', 'supplement', figure_s2),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['SOURCE_DATE_EPOCH'] = SOURCE_DATE_EPOCH
    for stem, label, _part, builder in FIGURES:
        fig = builder()
        align_panel_letters(fig)
        fig.savefig(OUT / f'{stem}.pdf', format='pdf', metadata={'CreationDate': None})
        fig.savefig(OUT / f'{stem}.eps', format='eps')
        fig.savefig(OUT / f'{stem}.png', format='png', dpi=600)
        w_mm, h_mm = (round(v * 25.4, 2) for v in fig.get_size_inches())
        plt.close(fig)
        print(f'{label:9s} {w_mm:6.1f} x {h_mm:5.1f} mm')


if __name__ == '__main__':
    main()
