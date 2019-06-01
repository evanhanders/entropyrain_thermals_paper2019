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

#aspect = [0.35, 0.35, 0.25, 0.25, 0.25, 0.25]
#CASES = [0.1, 0.5, 1, 2, 3, 4]
#ROOT_DIR='../weird_form_good_2D_runs/'
aspect = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
CASES = [0.1, 0.5, 1, 2, 3, 4, 5, 6]
ROOT_DIR='../'#good_2D_runs/z_bot_zero/'
DIRS=['{:s}AN_2D_thermal_nrho{}_Re6e2_Pr1_aspect{}_Lz20/'.format(ROOT_DIR, nrho, ar) for nrho, ar in zip(CASES, aspect)] 

CASES_3D = [0.5, 1, 2]
AR_3D    = [0.5, 0.4, 0.35]
THREED_DIR = '../good_3D_runs/'
DIRS_3D=['{:s}FC_3D_thermal_Re6e2_Pr1_eps1.00e-4_nrho{}_aspect{}/'.format(THREED_DIR,nrho, ar) for nrho, ar in zip(CASES_3D, AR_3D)] 
dict_3D = {}
for i in range(len(CASES_3D)):
    dict_3D[CASES_3D[i]] = DIRS_3D[i]

height, width = 2.5, 3.25 

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(width, height*2))
cax = plt.subplot(gs.new_subplotspec((30,   0), 50, 1000))
ax1 = plt.subplot(gs.new_subplotspec((100,  0), 380, 1000))
ax2 = plt.subplot(gs.new_subplotspec((620,  0), 380, 1000))

norm = matplotlib.colors.Normalize(vmin=0.8, vmax=len(CASES))
sm   = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])

for i, direc in enumerate(DIRS):
    f = h5py.File('{:s}/thermal_analysis/final_outputs.h5'.format(direc), 'r')
    ft = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(direc), 'r')
    for j in range(2):
        if j == 0:
            x, y     = ft['times'].value, f['d_measured'].value
            x_t, y_t = ft['times'].value, f['d_theory'].value
            ax = ax1
        elif j == 1:
            x, y     = f['d_measured'].value[2:-2], -f['w_measured'].value
            x_t, y_t = f['d_theory'].value[2:-2],   -f['w_theory'].value
            ax = ax2
        color = sm.to_rgba(i+1)
        ax.plot(x[:], y[:], marker='o', lw=0, markersize=3, markeredgecolor=(*color[:-1], 0.8), markerfacecolor=(*color[:-1], 0.2), markeredgewidth=0.5)
        if CASES[i] in CASES_3D:
            f_3Dt = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(dict_3D[CASES[i]]), 'r')
            f_3D = h5py.File('{:s}/thermal_analysis/final_outputs.h5'.format(dict_3D[CASES[i]]), 'r')
            if j == 0:
                x_3D, y_3D     = f_3Dt['times'].value, f_3D['d_measured'].value
            elif j == 1:
                x_3D, y_3D     = f_3D['d_measured'].value[2:-2], -f_3D['w_measured'].value
            f_3D.close()
            f_3Dt.close()
            ax.plot(x_3D[:], y_3D[:], marker='+', lw=0, markersize=2, markeredgecolor='k', markeredgewidth=0.5)
        ax.plot(x_t, y_t, c='k', lw=0.5)#c=color, lw=1)

        if j == 0:
            ax.set_xlim(0, 50)
            ax.set_yticks((5, 10, 15, 20))
            ax.set_ylim(2, 20)
            ax.set_ylabel('Depth')
            ax.set_xlabel('Time')
        elif j == 1:
            ax.set_xlim(0, 20)
            ax.set_ylim(0.2, 1.15)
            ax.set_yticks((0.5, 1))
            ax.set_ylabel('|w|')
            ax.set_xlabel('Depth')

    f.close()
    ft.close()
ax1.plot([100, 101], [100, 101], lw=0, marker='o', c='k', markersize=3, markerfacecolor=(0,0,0,0.2), markeredgecolor=(0,0,0,0.8), label='2D AN', markeredgewidth=0.5)
ax1.plot([100, 101], [100, 101], lw=0, marker='+', c='k', markersize=2, label='3D FC', markeredgewidth=0.5)
ax1.legend(loc='lower right', frameon=False, fontsize=8, handletextpad=0)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal', boundaries=np.linspace(1, len(CASES)+1, len(CASES)+1)-0.5, ticks=np.arange(len(CASES)+1))
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_ticklabels(CASES)
cb.set_label(r'$n_\rho$', labelpad=-40)

fig.savefig('results_panels.png', dpi=300, bbox_inches='tight')
