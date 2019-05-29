import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
matplotlib.rcParams.update({'font.size': 9})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

NRUNS=8

aspect = [0.25, 0.25, 0.25]
CASES = [0.5, 1, 2]
ROOT_DIR='../good_2D_runs/z_bot_zero/'
DIRS=['{:s}AN_2D_thermal_nrho{}_Re6e2_Pr1_aspect{}_Lz20/'.format(ROOT_DIR, nrho, ar) for nrho, ar in zip(CASES, aspect)] 

aspect_2D = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.25]
CASES_2D = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
color_dir = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS_2D=['{:s}AN_2D_thermal_nrho{}_Re6e2_Pr1_aspect{}_Lz20/'.format(ROOT_DIR, nrho, ar) for nrho, ar in zip(CASES_2D, aspect_2D)] 

CASES_3D = [0.5, 1, 2]
AR_3D    = [0.5, 0.4, 0.35]
THREED_DIR = '../good_3D_runs/'
DIRS_3D=['{:s}FC_3D_thermal_Re6e2_Pr1_eps1.00e-4_nrho{}_aspect{}/'.format(THREED_DIR,nrho, ar) for nrho, ar in zip(CASES_3D, AR_3D)] 
dict_3D = {}
for i in range(len(CASES_3D)):
    dict_3D[CASES_3D[i]] = DIRS_3D[i]


height, width = 2.5, 3.25 

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(width, height))

subplots = []
pad = 100
p_width = 1000
p_height = int(np.floor(950)/3)
subplots = [( (50, 0),                    p_height*2, p_width),
            ( (50+p_height*2, 0),         p_height,   p_width),
            ]

fig = plt.figure(figsize=(width, height))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
cax = plt.subplot(gs.new_subplotspec((0,   0), 50, 1000))

norm = matplotlib.colors.Normalize(vmin=0.8, vmax=NRUNS)
sm   = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])

for i, direc in enumerate(DIRS_2D):
    f = h5py.File('{:s}/thermal_analysis/final_outputs.h5'.format(direc), 'r')
    x, y     = f['d_measured'].value, f['r_measured'].value
    x_t, y_t = f['d_theory'].value,   f['r_theory'].value
    f.close()
    color = sm.to_rgba(color_dir[i])
    axs[0].plot(x, y, marker='o', lw=0, markersize=3, markeredgecolor=(*color[:-1], 0.8), markerfacecolor=(*color[:-1], 0.2), markeredgewidth=0.5)
    axs[0].plot(x_t, y_t, c='k', lw=0.5)#c=color, lw=1)



for i, direc in enumerate(DIRS):
    f = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(direc), 'r')
    f_3D = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(dict_3D[CASES[i]]), 'r')
    t, d, r     = f['times'].value, 20 - f['vortex_height'].value, f['vortex_radius'].value
    t3, d3, r3  = f_3D['times'].value, 20 - f_3D['vortex_height'].value, f_3D['vortex_radius'].value
    f.close()
    f_3D.close()


    color = sm.to_rgba(i+2)
    x2, y2, x3, y3 = d, r, d3, r3
    l = np.min((len(y2), len(y3)))
#    axs[0].plot(x2[:l], y2[:l], marker='o', lw=0, markersize=3, markeredgecolor=(*color[:-1], 0.8), markerfacecolor=(*color[:-1], 0.2), markeredgewidth=0.5)
    axs[0].plot(x3[:l], y3[:l], marker='+', lw=0, markersize=2, markeredgecolor='k', markerfacecolor='k', markeredgewidth=0.5)

    axs[1].axhline(1e-2, c='k', lw=0.25)
    axs[1].axhline(10**(-1.5), c='k', lw=0.25)
    axs[1].axhline(10**(-2.5), c='k', lw=0.25)

    diff = (1 - y2/y3)[:l]
    pos  = diff > 0
    neg  = diff < 0
    axs[1].plot(x2[:l][pos], diff[pos], markerfacecolor=color, markeredgecolor=color, marker='o', lw=0, markersize=2, markeredgewidth=0.35)
    axs[1].plot(x2[:l][neg], -diff[neg], markerfacecolor=(*color[:-1], 0), markeredgecolor=color, marker='o', lw=0, markersize=2, markeredgewidth=0.35)
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-3, 1e-1)
    axs[1].set_xlim(x2.min(), x2.max())

axs[0].set_ylim(0.2, 2)
axs[0].set_xlim(2, 20)
axs[1].set_xlim(2, 20)
axs[0].set_ylabel('Radius')
axs[1].set_ylabel(r'$1 - \frac{\mathrm{AN}}{\mathrm{FC}}$')
axs[1].set_xlabel('Depth')
axs[0].set_yscale('log')
axs[0].tick_params(labelbottom=False)
axs[0].set_yticks((0.3, 0.6, 1, 2))
axs[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axs[0].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


axs[0].plot([100, 101], [100, 101], lw=0,markersize=3,  marker='o', c='k', markerfacecolor=(1,1,1,0.8), markeredgecolor='k', label='2D AN', markeredgewidth=0.5)
axs[0].plot([100, 101], [100, 101], lw=0,markersize=2,  marker='+', c='k', label='3D FC', markeredgewidth=0.5)
#axs[0].plot([100, 101], [100, 101], lw=0.5, label='theory', c='k')
axs[0].legend(loc='upper left', frameon=False, fontsize=8, handletextpad=0.5, handlelength=0.8)
axs[1].plot([100, 101], [100, 101], lw=0,    markersize=3,  marker='o', c='k', markerfacecolor=(0,0,0,0), markeredgecolor=(0,0,0,1), label='< 0', markeredgewidth=0.35)
axs[1].plot([100, 101], [100, 101], lw=0,    markersize=3,  marker='o', c='k', markerfacecolor=(0,0,0,1), markeredgecolor=(0,0,0,1), label='> 0', markeredgewidth=0.35)
axs[1].legend(loc='upper center', frameon=False, fontsize=8, handletextpad=0, borderpad=0, ncol=2, borderaxespad=0.3)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal', boundaries=np.linspace(1, NRUNS+1, NRUNS+1), ticks=np.arange(NRUNS+1) + 0.5)
cax.xaxis.set_ticks_position('top')
#cax.xaxis.set_ticklabels([0.1, 0.5, 1, 2, 3, 4])
cax.xaxis.set_ticklabels([0.1, 0.5, 1, 2, 3, 4, 5, 6])
cb.set_label(r'$n_\rho$', labelpad=-30)


fig.savefig('diff_AN_FC.png', dpi=300, bbox_inches='tight')
