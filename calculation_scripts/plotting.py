import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.ticker as ticker
from scipy import ndimage
import os

def _plot_series(x, y, title, xlabel, ylabel, **kwargs):
    plt.figure()
    plt.plot(x, y, **kwargs)
    plt.title(title, fontsize=18)
    plt.ylabel(xlabel, fontsize=14)
    plt.xlabel(ylabel, fontsize=14)


def plotObjectiveFunction(max_iterations, data_path, plot_path, **kwargs):
    obj = np.loadtxt(os.path.join(data_path, "Obj.txt"))
    iterations = np.arange(max_iterations)
    _plot_series(iterations, obj, "Quality Function", "Quality", "Iteration", **kwargs)
    plt.savefig(os.path.join(plot_path, "obj.png"), bbox_inches='tight')


def plotStepSizes(max_iterations, data_path, plot_path, **kwargs):
    stepsizes = np.loadtxt(os.path.join(data_path, "stepsizes.txt"))
    iterations = np.arange(max_iterations)
    _plot_series(iterations, stepsizes, "Step Size", "Step Size", "Iteration", **kwargs)
    plt.savefig(os.path.join(plot_path, "stepsizes.png"), bbox_inches='tight')

def plotGeometry(all_parameters, d, plot_path, iteration, angle1=225, angle2=45, fill_zeros=True):
    num_x, num_y, num_z = all_parameters.shape
    X, Y, Z = d*np.indices((num_x+1, num_y+1, num_z+1))
    filled = np.ones((num_x, num_y, num_z))
    if not fill_zeros:
        filled = (all_parameters > 0.2)

    color_normalizer = matplotlib.colors.Normalize(0, 1)
    colorset = cm.ScalarMappable(norm=color_normalizer, cmap='Spectral_r')
    colors = cm.Spectral_r(color_normalizer(all_parameters))

    # Create a figure with a 3D projection, and plot the voxels.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolors='white', linewidth=0.2)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")

    #TODO: Find a way to scale z-axis properly so that empty plotting region is neglected.
    # Current method is hacky, and below are a couple ways that could work.
    #plt.gca().set_zscale(num_z/num_x, 'linear')
    #scale_z = num_z / num_x
    #ticks_z = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_z))
    #ax.zaxis.set_major_formatter(ticks_z)

    ax.set_zlim(0, d*num_x)     # hacky because depends on value of num_x (what is num_x < num_z?)
    ax.grid(False)
    ax.view_init(azim=angle1, elev=angle2)
    fig.colorbar(colorset, shrink=0.9, aspect=10, cax=ax.inset_axes([1.0, 0, 0.05, 0.8]))
    fig.suptitle(f'Structure at iteration {iteration}')
    plt.savefig(os.path.join(plot_path, f"Structure{iteration}.png"), dpi=100)


def EField_slice(E_tot, plot_path, iteration, index=0, axis='x', cbar_limits=None):
    E_tot_abs = np.absolute(E_tot)
    match axis:
        case 'x':
            E_plot = E_tot_abs[index,:,:]
        case 'y':
            E_plot = E_tot_abs[:,index,:]
        case 'z':
            E_plot = E_tot_abs[:,:,index]
    E_plot = ndimage.rotate(E_plot, 90)
    plt.figure()
    plt.imshow(E_plot, cmap='jet', interpolation='bilinear')
    #TODO: add axes and confirm orientation of plots. currently, one axis looks flipped.
    plt.axis('off')
    if cbar_limits:
        plt.clim(cbar_limits[0], cbar_limits[1])
    plt.colorbar()
    plt.title(f'Field Enhancement at {axis}={index}, iteration {iteration}')
    plt.savefig(os.path.join(plot_path, f"EField{iteration}_{axis}={index}.png"), dpi=100)
