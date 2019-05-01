import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

dirs = [    ('../AN_2D_thermal_nrho0.5_Re6e2_Pr1_aspect0.25_Lz20', (1, 0), (2, 18), (5, 17)) ,
            ('../apr30/AN_2D_thermal_nrho3_Re6e2_Pr1_aspect0.125_Lz20' , (1,0), (2, 12),  (4, 5))]


gs = gridspec.GridSpec(1000, 1000)

subplots = [    ( (50, 0),    400, 150 ),  ( (50, 150),  400, 150 ),
                ( (50, 350),  400, 150 ),  ( (50, 500),  400, 150 ),
                ( (50, 700),  400, 150 ),  ( (50, 850),  400, 150 ),
                ( (550, 0),   400, 150 ),  ( (550, 150), 400, 150 ),
                ( (550, 350), 400, 150 ),  ( (550, 500), 400, 150 ),
                ( (550, 700), 400, 150 ),  ( (550, 850), 400, 150 )
            ]

fig = plt.figure(figsize=(7.5, 5))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]

plot_count = 0
for i, ax in enumerate(axs):
    dir_ind = int(i % 2)
    this_dir = dirs[dir_ind][0]
    filenum  = dirs[dir_ind][1 + (plot_count % 3)][0]
    imgnum   = dirs[dir_ind][1 + (plot_count % 3)][1]
    print(filenum)
    f  = h5py.File('{:s}/slices/slices_s{}.h5'.format(this_dir, filenum), 'r')
    cf = h5py.File('{:s}/thermal_analysis/contour_file.h5'.format(this_dir), 'r')

    r = f['scales']['r']['1.0'].value
    z = f['scales']['z']['1.0'].value
    rr, zz = np.meshgrid(r.flatten(), z.flatten())
    t = f['scales']['sim_time'].value[imgnum]

    contour = cf['contours'].value[20*(filenum-1)+imgnum,:]

    if int(plot_count / 3) < 1:
        field = f['tasks']['w'].value[imgnum,:]
    else:
        field = f['tasks']['S1'].value[imgnum,:]
    f.close()
    cf.close()

    minval = field.min()

    ax.pcolormesh( rr, zz, field, cmap='RdBu_r', rasterized=True, vmin=minval, vmax=-minval)
    ax.pcolormesh(-rr, zz, field, cmap='RdBu_r', rasterized=True, vmin=minval, vmax=-minval)
    good = contour > 0
    if np.sum(good) > 0:
        ax.plot( contour[good], z.flatten()[good],  c='k', lw=0.5)
        ax.plot(-contour[good], z.flatten()[good],  c='k', lw=0.5)
        connectors_x1 = [contour[good][0], -contour[good][0]]
        connectors_y1 = [z.flatten()[good][0], z.flatten()[good][0]]
        connectors_x2 = [contour[good][-1], -contour[good][-1]]
        connectors_y2 = [z.flatten()[good][-1], z.flatten()[good][-1]]
        ax.plot(connectors_x1, connectors_y1,  c='k', lw=0.5)
        ax.plot(connectors_x2, connectors_y2,  c='k', lw=0.5)


    x_max = 5
    ax.text(-0.93*x_max, 0.5, r'$t = {:.1f}$'.format(t))
    if plot_count < 3:
        ax.text(2*x_max/3, 18, r'$w$')
    else:
        ax.text(2*x_max/3, 18, r'$\frac{s_1}{c_P}$')
    if plot_count == 0 or plot_count == 3:
        if dir_ind == 1:
            ax.text(-0.93*x_max, 2.5, r'$n_\rho = 3$')
        else:
            ax.text(-0.93*x_max, 2.5, r'$n_\rho = 0.5$')
    if dir_ind == 1:
        ax.set_xlim(-x_max, x_max)
        ax.set_yticklabels([])
        ax.set_xticks((0, x_max))
        plot_count += 1
    else:
        ax.set_xlim(-x_max, x_max)
        ax.set_xticks((-x_max, 0, x_max))
        ax.set_yticks((0, 5, 10, 15, 20))


fig.savefig('evolution_colormeshes.png', dpi=300)
