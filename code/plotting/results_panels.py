import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

CASES = [0.1, 0.5, 1, 2, 3, 4]
r_bounds = [(0, 2), (0, 2), (0, 1.5), (0, 1), (0, 3./4), (0, 3./4)]
ROOT_DIR='../may03/'
DIRS=['{:s}AN_2D_thermal_nrho{}_Re6e2_Pr1_aspect0.25_Lz20/'.format(ROOT_DIR,case) for case in CASES] 

height, width = 5, 6.5
pad = 50
h_pad = 100
p_height = int(np.round((1000 - 2*pad)/len(CASES)))
p_width  = int(np.round((1000 - 2*pad - h_pad)/3))

gs = gridspec.GridSpec(1000, 1000)

subplots = []
for i in range(len(CASES)):
    for j in range(3):
        if j > 0:
            hpad = h_pad + pad
        else:
            hpad = pad
        subplots.append( ( (pad+i*p_height, hpad+j*p_width), p_height, p_width))

fig = plt.figure(figsize=(width, height))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]


for i, direc in enumerate(DIRS):
    f = h5py.File('{:s}/thermal_analysis/final_outputs.h5'.format(direc), 'r')
    for j in range(3):
        ax = axs[i*3+j]
        if j == 0:
            x, y     = f['times'].value, f['d_measured'].value
            x_t, y_t = f['times'].value, f['d_theory'].value
        elif j == 1:
            x, y     = f['d_measured'].value, f['r_measured'].value
            x_t, y_t = f['d_theory'].value,   f['r_theory'].value
        elif j == 2:
            x, y     = f['d_measured'].value[2:-2], f['w_measured'].value
            x_t, y_t = f['d_theory'].value[2:-2],   f['w_theory'].value
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.plot(x, y, c='k', marker='o', lw=0, markersize=1, markeredgecolor='black')
        ax.plot(x_t, y_t, c='orange', lw=1)

        if j == 0:
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 20)
            ax.set_ylabel('depth')
            ax.text(0.5, 16, r'$n_\rho = {}$'.format(CASES[i]), size=10)
        elif j == 1:
            ax.set_xlim(0, 20)
            ax.set_yticks((0.5, 1, 1.5, 2))
            ax.set_ylim(*r_bounds[i])
            ax.set_ylabel('r')
        elif j == 2:
            ax.set_xlim(0, 20)
            ax.set_ylim(-1, 0)
            ax.set_yticks((-1, -0.5))
            ax.set_ylabel('w')

        if i < len(DIRS) - 1:
            ax.tick_params(labelbottom=False)
        else:
            if j == 0:
                ax.set_xlabel('t')
            else:
                ax.set_xlabel('depth')


    f.close()

fig.savefig('results_panels.png', dpi=300, bbox_inches='tight')
