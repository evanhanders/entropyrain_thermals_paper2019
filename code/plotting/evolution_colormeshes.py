import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
matplotlib.rcParams.update({'font.size': 9})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import numpy as np
import h5py

dirs = [    ('../AN_2D_thermal_nrho0.5_Re6e2_Pr1_aspect0.25_Lz20', (1, 0), (2, 18), (5, 17)) ,
            ('../AN_2D_thermal_nrho3_Re6e2_Pr1_aspect0.25_Lz20' , (1,0), (2, 12),  (4, 5))]


gs = gridspec.GridSpec(1000, 1000)

subplots = [    ( (100, 50),    900, 450 ),  ( (100, 550),  900, 450 ) ]
colorbar = [    ( (50, 50),    50, 450 ),  ( (50, 550),  50, 450 ) ]

fig = plt.figure(figsize=(3.25, 3))
axs = [plt.subplot(gs.new_subplotspec(*args)) for args in subplots]
cax = plt.subplot(gs.new_subplotspec( (50, 50),    50, 950 ))

plot_count = 0
for i, ax in enumerate(axs):
    this_dir = dirs[i][0]
    if i == 0:
        B = -0.568
    else:
        B = -0.946
    for j, info in enumerate(dirs[i][1:]):

        filenum, imgnum = info
        print('opening file ', this_dir, filenum, imgnum)
        f  = h5py.File('{:s}/slices/slices_s{}.h5'.format(this_dir, filenum), 'r')
        cf = h5py.File('{:s}/thermal_analysis/contour_file.h5'.format(this_dir), 'r')
        ff = h5py.File('{:s}/thermal_analysis/final_outputs.h5'.format(this_dir), 'r')
        pf = h5py.File('{:s}/thermal_analysis/post_analysis.h5'.format(this_dir), 'r')

        r = f['scales']['r']['1.0'].value
        z = f['scales']['z']['1.0'].value
        rr, zz = np.meshgrid(r.flatten(), z.flatten())
        t = f['scales']['sim_time'].value[imgnum]

        volumes = pf['int_vol'].value
        this_volume = volumes[(filenum-1)*20+imgnum]

        n_rho = float(this_dir.split('_nrho')[-1].split('_Re')[0])
        grad_T = -(np.exp(n_rho/1.5)-1)/20
        rho = (1 + grad_T*(zz-20))**1.5
        rho0 = (1 + grad_T*(18.5-20))**1.5

        contour = cf['contours'].value[20*(filenum-1)+imgnum,:]
        contours = cf['contours'].value 
        heights  = 20 - ff['d_measured'].value
        radius   = ff['r_measured'].value

        this_r = radius[20*(filenum-1) + imgnum]
        torus_r = this_r/5

        print(np.max(contour))
        if j > 0:
            if i == 0:
                #Estimate of torus volume
                rho *= 2*np.pi**2 * this_r**3 / 4**2 /np.abs(B)
            else:
                rho *= 2*np.pi**2 * this_r**3 / 3**2 /np.abs(B)
#            rho *= this_volume/np.abs(B)
        else:
            rho *= (4./3)*np.pi*(0.5)**3/ np.abs(B)
        
        field = f['tasks']['S1'].value[imgnum,:]
        f.close()
        ff.close()
        cf.close()

        minval = -1
        breaks = len(dirs[i]) - 1
        good   = (zz > (breaks-1-j)/breaks * 20)*(zz <= (breaks-j)/breaks * 20)
        z_shape = np.sum((z > (breaks-1-j)/breaks * 20)*(z <= (breaks-j)/breaks * 20))
        ax.pcolormesh( rr[good].reshape(z_shape, len(r)), zz[good].reshape(z_shape, len(r)), (rho*field)[good].reshape(z_shape, len(r)), cmap='Blues_r', rasterized=True, vmin=minval, vmax=0)
        c = ax.pcolormesh(-rr[good].reshape(z_shape, len(r)), zz[good].reshape(z_shape, len(r)), (rho*field)[good].reshape(z_shape, len(r)), cmap='Blues_r', rasterized=True, vmin=minval, vmax=0)
        if i == 0 and j == 0:
            bar = plt.colorbar(c, cax=cax, orientation='horizontal')
            bar.solids.set_rasterized(True)
            cax.xaxis.set_ticks_position('top')
            bar.set_label(r'$\frac{\rho S_1 \mathcal{V}}{B_{\mathrm{th}}}$', labelpad=-38)
        x_max = 5
        if j == len(dirs[i])-2:
            max_contours = contours.max(axis=1)
            good_a = (max_contours > 0)*(heights <= 14)
            good_a[2] = True

            max_contours = max_contours[good_a]
            zs_contours = heights[good_a]
            contour_f = interp1d(zs_contours, max_contours, bounds_error=False, fill_value='extrapolate')
            ax.plot( contour_f(zs_contours[:-1]), zs_contours[:-1], c='k', lw=0.25)
            ax.plot(-contour_f(zs_contours[:-1]), zs_contours[:-1], c='k', lw=0.25)
         
            heights = [17, 14, 11, 8, 5, 2]
            dy = -0.5
            for h in heights:
                this_r = contour_f(h)
                next_r = contour_f(h+dy)
                dx = next_r-this_r
                ax.arrow( this_r, h,   dx,  dy, shape='full', lw=0, length_includes_head=True, head_width=0.25, color='black')
                ax.arrow(-this_r, h, -(dx), dy, shape='full', lw=0, length_includes_head=True, head_width=0.25, color='black')
        good_c = contour > 0
        if np.sum(good_c) > 0:
            ax.plot( contour[good_c], z.flatten()[good_c],  c='k', lw=0.25)
            ax.plot(-contour[good_c], z.flatten()[good_c],  c='k', lw=0.25)
            connectors_x1 = [contour[good_c][0], -contour[good_c][0]]
            connectors_y1 = [z.flatten()[good_c][0], z.flatten()[good_c][0]]
            connectors_x2 = [contour[good_c][-1], -contour[good_c][-1]]
            connectors_y2 = [z.flatten()[good_c][-1], z.flatten()[good_c][-1]]
            ax.plot(connectors_x1, connectors_y1,  c='k', lw=0.25)
            ax.plot(connectors_x2, connectors_y2,  c='k', lw=0.25)


        ax.set_yticks((0, 5, 10, 15, 20))
        if i == 1:
            ax.set_xlim(-x_max, x_max)
            ax.set_yticklabels([])
            ax.set_xticks((0, x_max))
            ax.text(-0.97*x_max, 18.5, r'$n_\rho = 3$', size=9)
#            ax.text(-4.7, 0.3, 'b', fontsize=12)
        else:
            ax.set_xlim(-x_max, x_max)
            ax.set_xticks((-x_max, 0, x_max))
            ax.text(-0.97*x_max, 18.5, r'$n_\rho = \frac{1}{2}$', size=9)
            ax.set_ylabel('z')
#            ax.text(-4.7, 0.3, 'a', fontsize=12)
        ax.set_xlabel('x')


fig.savefig('evolution_colormeshes.png', dpi=300, bbox_inches='tight')
fig.savefig('evolution_colormeshes.pdf', dpi=600, bbox_inches='tight')
