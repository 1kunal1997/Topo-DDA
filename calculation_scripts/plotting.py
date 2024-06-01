import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import ndimage
import os

def _plot_series(x, y, title, xlabel, ylabel, **kwargs):
    plt.figure()
    plt.plot(x, y, **kwargs)
    plt.title(title, fontsize=22)
    plt.ylabel(xlabel, fontsize=18)
    plt.xlabel(ylabel, fontsize=18)


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


def plotPenaltyCoefficients(max_iterations, data_path, plot_path, **kwargs):
    coeffs=np.loadtxt(os.path.join(data_path, "Coeffs.txt"))
    iterations = np.arange(max_iterations)
    _plot_series(iterations, coeffs, "Penalty Weight", "Coefficient", "Iteration", **kwargs)
    plt.savefig(os.path.join(plot_path, "Coefficients.png"), bbox_inches='tight')


def plotAveragePenalty(max_iterations, data_path, plot_path, **kwargs):
    penalties=np.loadtxt(os.path.join(data_path, "Average_Penalties.txt"))
    iterations = np.arange(max_iterations)
    _plot_series(iterations, penalties, "Average Penalty", "Penalty", "Iteration", **kwargs)
    plt.savefig(os.path.join(plot_path, "average_penalty.png"), bbox_inches='tight')


def plotPenalties(x, penalty_shape, data_path, plot_path, iteration):
    params=np.load(os.path.join(data_path, "Parameters", f"Param{iteration}.npy"))
    penalties = np.load(os.path.join(data_path, "Penalties", f"Penalty{iteration}.npy"))
    plt.plot(x, penalty_shape)
    plt.plot(params, penalties, 'o')
    plt.title("Penalty Function")
    plt.ylabel("Penalty")
    plt.xlabel("Pixel Value")
    plt.savefig(os.path.join(plot_path, "Penalties", f"penalty{iteration}.png"), bbox_inches='tight')
    plt.close()


def plotPenaltyGradients(x, penalty_gradient_shape, data_path, plot_path, iteration):
    params=np.load(os.path.join(data_path, "Parameters", f"Param{iteration}.npy"))
    gradients_penalty = np.load(os.path.join(data_path, "Gradient_Penalties", f"Gradient{iteration}.npy"))
    plt.plot(x, penalty_gradient_shape)
    plt.plot(params, gradients_penalty, 'o')
    plt.title(f"Gradient of Penalty at Iteration {iteration}")
    plt.ylabel("Gradient")
    plt.xlabel("Pixel Value")
    plt.savefig(os.path.join(plot_path, "Gradients_Penalty", f"penaltyGradient_{iteration}.png"), bbox_inches='tight')
    plt.close()


def plotGeometry(all_parameters, plot_path, iteration, angle1=225, angle2=45, fill_zeros=True):
    num_x, num_y, num_z = all_parameters.shape
    X, Y, Z = np.indices((num_x+1, num_y+1, num_z+1))
    filled = np.ones((num_x, num_y, num_z))
    if not fill_zeros:
        filled = (all_parameters > 0.2)

    color_normalizer = matplotlib.colors.Normalize(0, 1)
    colorset = cm.ScalarMappable(norm=color_normalizer, cmap='Spectral_r')
    colors = cm.Spectral_r(color_normalizer(all_parameters))

    # Create a figure with a 3D projection, and plot the voxels.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolor='white', linewidth=0.2)
    # ax.grid(False)
    ax.view_init(azim=angle1, elev=angle2)
    fig.colorbar(colorset, shrink=0.9, aspect=10, cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]))
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
    plt.axis('off')
    if cbar_limits:
        plt.clim(cbar_limits[0], cbar_limits[1])
    plt.colorbar()
    plt.savefig(os.path.join(plot_path, f"EField{iteration}_{axis}={index}.png"), dpi=100)
