import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

aspect = [0.35, 0.25, 0.25]
CASES = [0.5, 1, 2]
TOFF  = [0.704, 1.09, 1.26]
ROOT_DIR='../good_2D_runs/'
DIRS=['{:s}AN_2D_thermal_nrho{}_Re6e2_Pr1_aspect{}_Lz20/'.format(ROOT_DIR, nrho, ar) for nrho, ar in zip(CASES, aspect)] 

CASES_3D = [0.5, 1, 2]
AR_3D    = [0.5, 0.4, 0.35]
TOFF_3D  = [0.583, 1.26, 1.53]
THREED_DIR = '../good_3D_runs/'
DIRS_3D=['{:s}FC_3D_thermal_Re6e2_Pr1_eps1.00e-4_nrho{}_aspect{}/'.format(THREED_DIR,nrho, ar) for nrho, ar in zip(CASES_3D, AR_3D)] 
dict_3D = {}
for i in range(len(CASES_3D)):
    dict_3D[CASES_3D[i]] = DIRS_3D[i]


height, width = 2.5, 6.5 

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(width, height))

subplots = []
pad = 100
p_width = int(np.floor(1000-1*pad)/2)
p_height = int(np.floor(1000)/3)
subplots = [( (0, 0),                    p_height*2, p_width),
            ( (0, pad+p_width),          p_height*2, p_width),
            ( (p_height*2, 0),           p_height,   p_width),
            ( (p_height*2, pad+p_width), p_height, p_width)
            ]

fig = plt.figure(figsize=(width, height))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]

norm = matplotlib.colors.Normalize(vmin=1, vmax=6)
sm   = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])

for i, direc in enumerate(DIRS):
    f = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(direc), 'r')
    f_3D = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(dict_3D[CASES[i]]), 'r')
    t, d, r     = f['times'].value, 20 - f['vortex_height'].value, f['vortex_radius'].value
    t3, d3, r3  = f_3D['times'].value, 20 - f_3D['vortex_height'].value, f_3D['vortex_radius'].value
    f.close()
    f_3D.close()


    color = sm.to_rgba(i+2)
    for j, x2, y2, x3, y3 in zip(range(2), (t, d), (d, r), (t3, d3), (d3, r3)):
        l = np.min((len(y2), len(y3)))
        axs[j].plot(x2[:l], y2[:l], marker='o', lw=0, markersize=3, markeredgecolor=(*color[:-1], 0.8), markerfacecolor=(*color[:-1], 0.2), markeredgewidth=0.5)
        axs[j].plot(x3[:l], y3[:l], marker='x', lw=0, markersize=2, markeredgecolor='k', markerfacecolor='k', markeredgewidth=0.5)#markeredgecolor=(*color[:-1], 0.9), markerfacecolor=(*color[:-1], 0.2))

        axs[j+2].axhline(1e-2, c='k', lw=0.25)
        axs[j+2].axhline(10**(-1.5), c='k', lw=0.25)
        axs[j+2].axhline(10**(-2.5), c='k', lw=0.25)

        diff = (1 - y2/y3)[:l]
        pos  = diff > 0
        neg  = diff < 0
        axs[j+2].plot(x2[:l][pos], diff[pos], markerfacecolor=color, markeredgecolor=color, marker='o', lw=0, markersize=2, markeredgewidth=0.35)
        axs[j+2].plot(x2[:l][neg], -diff[neg], markerfacecolor=(*color[:-1], 0), markeredgecolor=color, marker='o', lw=0, markersize=2, markeredgewidth=0.35)
        axs[j+2].set_yscale('log')
        axs[j+2].set_ylim(1e-3, 1e-1)
        axs[j+2].set_xlim(x2.min(), x2.max())

axs[0].set_xlim(0, 50)
axs[2].set_xlim(0, 50)
axs[0].set_ylim(0, 20)
axs[1].set_ylim(0.3, 1.6)
for i in (1, 3):
    axs[i].set_xlim(2, 20)
axs[0].set_ylabel('depth')
axs[1].set_ylabel('r')
axs[2].set_ylabel('w')
axs[2].set_ylabel(r'$1 - \frac{\mathrm{AN}}{\mathrm{FC}}$')
axs[2].set_xlabel(r'$t/t_b$')
axs[3].set_xlabel('depth')

axs[0].set_yticks([5, 10, 15, 20])
axs[0].tick_params(labelbottom=False)
axs[1].tick_params(labelbottom=False)

axs[0].plot([100, 101], [100, 101], lw=0,markersize=3,  marker='o', c='k', markerfacecolor=(1,1,1,0.8), markeredgecolor='k', label='2D AN')
axs[0].plot([100, 101], [100, 101], lw=0,markersize=2,  marker='x', c='k', label='3D FC')
axs[0].legend(loc='lower right', frameon=False, fontsize=8, handletextpad=0)
axs[2].plot([100, 101], [100, 101], lw=0,    markersize=3,  marker='o', c='k', markerfacecolor=(0,0,0,0), markeredgecolor=(0,0,0,1), label='< 0')
axs[2].plot([100, 101], [100, 101], lw=0,    markersize=3,  marker='o', c='k', markerfacecolor=(0,0,0,1), markeredgecolor=(0,0,0,1), label='> 0')
axs[2].legend(loc='upper left', frameon=False, fontsize=8, handletextpad=0, borderpad=0, ncol=2)

#cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
#cax.xaxis.set_ticks_position('top')
#cax.xaxis.set_ticklabels(CASES)
#cb.set_label(r'$n_\rho$', labelpad=-40)
fig.savefig('diff_AN_FC.png', dpi=300, bbox_inches='tight')
