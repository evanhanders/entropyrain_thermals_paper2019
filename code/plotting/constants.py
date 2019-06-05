import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
#matplotlib.rcParams['mathtext.cal'] = 'DejaVu Sans:caligraphic'
#matplotlib.rcParams['mathtext.fallback_to_cm'] = True 
#matplotlib.rcParams.update({'font.size': 9})
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

height, width = 4, 3.25 

gs = gridspec.GridSpec(1000, 1000)
fig = plt.figure(figsize=(width, height))

subplots = []
p_width = 1000#int(np.floor(1000-2*pad)/3)
pad = 30
p_height = int(np.floor((850-2*pad)/3))
for j in range(3):
    subplots.append( ( (70+j*(p_height+pad), 0), p_height, p_width))

fig = plt.figure(figsize=(width, height))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
cax = plt.subplot(gs.new_subplotspec(*((0, 0), 50, 1000)))

norm = matplotlib.colors.Normalize(vmin=0.8, vmax=len(DIRS))
sm   = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])

for i, direc in enumerate(DIRS):
    f = h5py.File('{:s}/thermal_analysis/post_analysis.h5'.format(direc), 'r')
    t, Gam, S, V     = f['times'].value, f['int_circ'].value, f['int_rho_s1'].value, f['int_vol'].value
    f.close()

    f = h5py.File('{:s}/thermal_analysis/z_cb_file.h5'.format(direc), 'r')
    r, d     = f['vortex_radius'].value, 20 - f['vortex_height'].value
    f.close()


    fit_f = h5py.File('{:s}/thermal_analysis/fit_file.h5'.format(direc), 'r')
    Gamma0, B0, f0 = fit_f['Gamma0'].value, fit_f['B0'].value, fit_f['f'].value
    fit_f.close()

    color = sm.to_rgba(i+1)
    good = S < 0.5*S.min()
    good[:10] = True
    V0 = V/r**3
    for j, x, y in zip(range(3), (d, d, d), (S/B0, Gam/Gamma0, V0/f0)):
        axs[j].plot(x[good], y[good], lw=0.75, c=color)

axs[0].set_ylim(0.75, 1.1)
axs[1].set_ylim(0.75, 1.1)
axs[2].set_ylim(0.75, 1.1)

for i in (0, 1, 2):
    axs[i].set_xlim(2, 20)
axs[2].set_ylabel('w')
axs[0].set_ylabel(r'$B/B_{\mathrm{th}}$')
axs[1].set_ylabel(r'$\Gamma/\Gamma_{\mathrm{th}}$')
axs[2].set_ylabel(r'$\mathcal{V}/(mr^3)$')
axs[2].set_xlabel('Depth')



axs[0].tick_params(labelbottom=False)
#axs[0].xaxis.set_ticks([])
#axs[1].xaxis.set_ticks([])
axs[1].tick_params(labelbottom=False)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal', boundaries=np.linspace(1, len(CASES)+1, len(CASES)+1)-0.5, ticks=np.arange(len(CASES)+1))
cb.solids.set_rasterized(True)
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_ticklabels(CASES)
cb.set_label(r'$n_\rho$', labelpad=-35)

#axs[0].text(19, 1.03, 'a', fontsize=12)
#axs[1].text(19, 1.03, 'b', fontsize=12)
#axs[2].text(19, 1.03, 'c', fontsize=12)



fig.savefig('constants.png', dpi=300, bbox_inches='tight')
fig.savefig('constants.pdf', dpi=600, bbox_inches='tight')
