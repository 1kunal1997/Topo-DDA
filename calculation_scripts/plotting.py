import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import ndimage
import os

def plotObjectiveFunction(max_iterations, data_path, full_path):
    #plt.figure(1)
    obj=np.loadtxt(os.path.join(data_path, "Obj.txt"))
    iterations = np.arange(max_iterations)
    plt.plot(iterations, obj)
    #plt.legend(loc='lower right')
    plt.title('Objective Function Plot')
    plt.ylabel('Objective Function')
    plt.xlabel('Iteration #')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(full_path, "obj.png"), bbox_inches='tight')
    plt.close()

def plotPenaltyCoefficients(max_iterations, data_path, plot_path):
    coeffs=np.loadtxt(os.path.join(data_path, "Coeffs.txt"))
    iterations = np.arange(max_iterations)
    plt.plot(iterations, coeffs)
    #plt.legend(loc='lower right')
    plt.title('Coefficients for Penalty')
    plt.ylabel('Coefficient')
    plt.xlabel('Iteration #')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "Coefficients.png"), bbox_inches='tight')
    plt.close()

def plotAveragePenalty(max_iterations, data_path, plot_path):
    penalties=np.loadtxt(os.path.join(data_path, "Penalties.txt"))
    iterations = np.arange(max_iterations)
    plt.plot(iterations, penalties)
    #plt.legend(loc='lower right')
    plt.title('Average Penalty of Structures')
    plt.ylabel('Average Penalty')
    plt.xlabel('Iteration #')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "average_penalty.png"), bbox_inches='tight')
    plt.close()

def plotPenalties(data_path, plot_path, iteration):
    x, penalty_shape=np.loadtxt(os.path.join(data_path, "Penalty_Shape.txt"), delimiter=',', unpack=True)
    params=np.loadtxt(os.path.join(data_path, "Parameters", f"Param{iteration}.txt"))
    penalties = np.loadtxt(os.path.join(data_path, "Penalties", f"Penalty{iteration}.txt"))
    plt.plot(x, penalty_shape)
    plt.plot(params, penalties, 'o')
    #plt.legend(loc='lower right')
    plt.title('Shape of Penalty')
    plt.ylabel('Penalty')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "Penalties", f"penalty{iteration}.png"), bbox_inches='tight')
    plt.close()

def plotPenaltyGradients(data_path, plot_path, iteration):
    x, penalty_gradient_shape=np.loadtxt(os.path.join(data_path, "PenaltyGradient_Shape.txt"), delimiter=',', unpack=True)
    params=np.loadtxt(os.path.join(data_path, "Parameters", f"Param{iteration}.txt"))
    gradients_penalty = np.loadtxt(os.path.join(data_path, "Gradients_Penalty", f"Gradient{iteration}.txt"))
    plt.plot(x, penalty_gradient_shape)
    plt.plot(params, gradients_penalty, 'o')
    #plt.legend(loc='lower right')
    plt.title('Shape of Gradient of Penalty')
    plt.ylabel('Gradient')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "Gradients_Penalty", f"penaltyGradient_{iteration}.png"), bbox_inches='tight')
    plt.close()

def plotParabolic(data_path, plot_path, info_dict):
    x, penalty_shape=np.loadtxt(os.path.join(data_path, "Penalty_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_shape)
    #plt.legend(loc='lower right')
    plt.title('Parabolic Penalty')
    plt.ylabel('Penalty')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penalty_shape.png"), bbox_inches='tight')
    plt.close()

    x, penalty_gradient_shape=np.loadtxt(os.path.join(data_path, "PenaltyGradient_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_gradient_shape)
    #plt.legend(loc='lower right')
    plt.title('Gradient of Parabolic Penalty')
    plt.ylabel('Gradient')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penaltyGradient_shape.png"), bbox_inches='tight')
    plt.close()

def plotGaussian(data_path, plot_path, info_dict):
    sigma = info_dict["sigma"]
    mu = info_dict["mu"]
    x, penalty_shape=np.loadtxt(os.path.join(data_path, "Penalty_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_shape)
    #plt.legend(loc='lower right')
    plt.title(f'Gaussian Penalty, Width={sigma}, Center at {mu}')
    plt.ylabel('Penalty')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penalty_shape.png"), bbox_inches='tight')
    plt.close()

    x, penalty_gradient_shape=np.loadtxt(os.path.join(data_path, "PenaltyGradient_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_gradient_shape)
    #plt.legend(loc='lower right')
    plt.title(f'Gradient of Gaussian Penalty, Width={sigma}, Center at {mu}')
    plt.ylabel('Gradient')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penaltyGradient_shape.png"), bbox_inches='tight')
    plt.close()

def plotTriangular(data_path, plot_path, info_dict):
    slope = info_dict["slope"]
    x, penalty_shape=np.loadtxt(os.path.join(data_path, "Penalty_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_shape)
    #plt.legend(loc='lower right')
    plt.title(f'Triangular Penalty, Slope={slope}')
    plt.ylabel('Penalty')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penalty_shape.png"), bbox_inches='tight')
    plt.close()

    x, penalty_gradient_shape=np.loadtxt(os.path.join(data_path, "PenaltyGradient_Shape.txt"), delimiter=',', unpack=True)
    plt.plot(x, penalty_gradient_shape)
    #plt.legend(loc='lower right')
    plt.title(f'Gradient of Triangular Penalty, Slope={slope}')
    plt.ylabel('Gradient')
    plt.xlabel('Pixel Value')
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.savefig(os.path.join(plot_path, "penaltyGradient_shape.png"), bbox_inches='tight')
    plt.close()

def Shape(geometry,diel,d,angle1=225,angle2=45,colormax=1,shapebarrier=0.5,plotDpi=100,iteration=-1,position="./",FullLattice=False):
    """Plot the shape of object as dot matrix.
    #Input:
    # --SC                                                         SolutionClass
    #   Solution Class.
    # --FullLattice   Boolean
    #   If true. Geometry is a full n*n*n matrix. diel=0 for point seen as air.
    """
    #d = SC.Get_d()
    N=round(np.shape(geometry)[0]/3)
    if(N!=np.shape(diel)[0]/3):
        print("size not equal!")
    geometry=np.reshape(geometry,(N,3))
    #geometry = SC.Get_geometry()
    for i in range(3):
        geometry[:,i] -= np.amin(geometry[:,i])
    [X,Y,Z] = map(int,list(np.amax(geometry,axis=0)+1))
    Axis_max = max(X,Y,Z)*1.2*d

    diel=np.reshape(diel,(N,3))
    diel=diel[:,0]
    #diel = SC.Get_diel()[:,0]

    cmaparg = 'Spectral_r'
    minn, maxx = 0, colormax
    norm = matplotlib.colors.Normalize(minn, maxx)
    colorset = cm.ScalarMappable(norm=norm, cmap=cmaparg)
    colorset.set_array([])
    if FullLattice:
        index = np.where(diel>shapebarrier)
        diel = diel[index]
        geometry = geometry[index]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    geo_dic = set()
    surf_list = []
    surf_color = []
    x_grid, y_grid, z_grid = (np.indices((X+1,Y+1,Z+1))-0.5)*d
    filled, colors = np.zeros((X,Y,Z)), np.zeros((X,Y,Z))
    for i,pos in enumerate(geometry):
        geo_dic.add((pos[0],pos[1],pos[2]))
    for i,pos in enumerate(geometry):
        if (pos[0]+1,pos[1],pos[2]) not in geo_dic or (pos[0]-1,pos[1],pos[2]) not in geo_dic or\
           (pos[0],pos[1]+1,pos[2]) not in geo_dic or (pos[0],pos[1]-1,pos[2]) not in geo_dic or\
           (pos[0],pos[1],pos[2]+1) not in geo_dic or (pos[0],pos[1],pos[2]-1) not in geo_dic:
            filled[pos[0],pos[1],pos[2]] = 1
            colors[pos[0],pos[1],pos[2]] = diel[i]
    surf_list = np.array(surf_list)
    surf_color = np.array(surf_color)
    # print(x_grid.shape,y_grid.shape,z_grid.shape,filled.shape,colors.shape)
    colors = cm.Spectral_r(norm(colors))
    ln=ax.voxels(x_grid,y_grid,z_grid,filled.astype(bool),facecolors=colors,edgecolor ='white',linewidth=0.2)
    #ax2.scatter(geometry[:,0]*d,geometry[:,1]*d,geometry[:,2]*d,c=E_tot_abs, s=15, cmap=cmaparg)
    ax.set_xlim(-(Axis_max-X*d)/2,(Axis_max+X*d)/2)
    ax.set_ylim(-(Axis_max-Y*d)/2,(Axis_max+Y*d)/2)
    ax.set_zlim(-(Axis_max-Z*d)/2,(Axis_max+Z*d)/2)
    ax.set_xlabel("x[nm]")
    ax.set_ylabel("y[nm]")
    ax.set_zlabel("z[nm]")
    ax.grid(False)
    ax.view_init(azim=angle1, elev=angle2)
    fig.colorbar(colorset, shrink=0.9, aspect=10, cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]))
     
    #plt.show()
    #plt.savefig("./optimization_geometries/Iteration{}.png".format(it))

    fig.suptitle(f"iteration{iteration}")
    if FullLattice:
        plt.savefig(os.path.join(position, "SolidStructures", f"Structure{iteration}.png"),dpi=plotDpi)
    else:
        plt.savefig(os.path.join(position, "Structures", f"Structure{iteration}.png"),dpi=plotDpi)
    #plt.show()
    plt.close()


def EField_slice(geometry, E_tot, d, plotDpi=100, Elimit=False, Elimitlow=-1,Elimithigh=-1,Xslice=-1,Yslice=-1,Zslice=-1,iteration=-1, position="./"):
    """Plot the E field of object as arrow matrix.
    # Input:
    # --SC         SolutionClass
    #   Solved Solution Class.
    # --idx1,idx2  int
    #   Indexs of the target instance.
    """
    #print(geometry)

    N=round(np.shape(geometry)[0]/3)
    #print(N)
    geometry=np.reshape(geometry,(N,3))
    
   # print(geometry.shape)
    #diel=np.reshape(diel,(N,3))
    #diel=diel[:,0]
    

    for i in range(3):
        geometry[:,i] -= np.amin(geometry[:,i])
    
    #print(geometry)
    [X,Y,Z] = list(np.amax(geometry,axis=0)+1)
    Axis_max = max(X,Y,Z)*1.2*d
    
   # print("{}, {}, {}".format(X,Y,Z))

    E_tot = E_tot.reshape(int(E_tot.size/3),3)
    E_tot_real=E_tot.real
    E_tot_imag=E_tot.imag
    E_tot_abs = np.sqrt(np.sum(E_tot_real**2+E_tot_imag**2,axis=1))
    
    slicedim=-1
    
    if Xslice!=-1:
        position = os.path.join(position, "E-Field_XSlice")
        slicedim=0
        Eslice=np.zeros((Y, Z))
    if Yslice!=-1:
        position = os.path.join(position, "E-Field_YSlice")
        slicedim=1
        Eslice=np.zeros((Z, X))
    if Zslice!=-1:
        position = os.path.join(position, "E-Field_ZSlice")
        slicedim=2
        Eslice=np.zeros((X, Y))
    
    slicepos= max([Xslice, Yslice, Zslice])
    for i, ele in enumerate(E_tot_abs):
        if geometry[i][slicedim]==slicepos:
            Eslice[geometry[i][slicedim-2]][geometry[i][slicedim-1]]=ele

    """
    if Xslice!=-1:
        slicedim=0
        Eslice=np.zeros((Z, Y))
    if Yslice!=-1:
        slicedim=1
        Eslice=np.zeros((X, Z))
    if Zslice!=-1:
        slicedim=2
        Eslice=np.zeros((Y, X))
    
    slicepos= max([Xslice, Yslice, Zslice])
    for i, ele in enumerate(E_tot_abs):
        if geometry[i][slicedim]==slicepos:
            Eslice[geometry[i][slicedim-1]][geometry[i][slicedim-2]]=ele
    """
    rotated_img = ndimage.rotate(Eslice, 90)
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(rotated_img, cmap='jet', interpolation='bilinear')
    plt.axis('off')
    if Elimit:
        plt.clim(Elimitlow, Elimithigh)
    plt.colorbar()
    plt.savefig(os.path.join(position, f"EField{iteration} E_slice_{(["X", "Y", "Z"])[slicedim]}at{slicepos}.png"), dpi=plotDpi) 
    plt.close()

def EField(geometry, k_dir, E_dir, E_tot, d, iteration=-1, position="./"):
    """Plot the E field of object as arrow matrix.
    # Input:
    # --SC         SolutionClass
    #   Solved Solution Class.
    # --idx1,idx2  int
    #   Indexs of the target instance.
    """
    #print(geometry)

    N=round(np.shape(geometry)[0]/3)
    print(N)
    geometry=np.reshape(geometry,(N,3))
    
    print(geometry.shape)
    #diel=np.reshape(diel,(N,3))
    #diel=diel[:,0]
    

    for i in range(3):
        geometry[:,i] -= np.amin(geometry[:,i])
    
    #print(geometry)
    [X,Y,Z] = list(np.amax(geometry,axis=0)+1)
    Axis_max = max(X,Y,Z)*1.2*d
    
    print("{}, {}, {}".format(X,Y,Z))

    E_tot = E_tot.reshape(int(E_tot.size/3),3)
    E_tot_abs = np.abs(np.sqrt(np.sum((E_tot)**2,axis=1)))
    
    """
    if FullLattice:
        index = np.where(diel!=0)                                      #disabled all codes with "diel"
        geometry = geometry[index]
        E_tot = E_tot[index]
        E_tot_abs = E_tot_abs[index]
    """
    cmaparg = 'Spectral_r'
    minn, maxx = E_tot_abs.min(), E_tot_abs.max()
    print(minn,maxx,np.argmax(E_tot_abs))
    norm = matplotlib.colors.Normalize(minn, maxx)
    colorset = cm.ScalarMappable(norm=norm, cmap=cmaparg)
    colorset.set_array([])
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.quiver(geometry[:,0]*d,geometry[:,1]*d,geometry[:,2]*d,np.real(E_tot[:,0]),np.real(E_tot[:,1]),np.real(E_tot[:,2]),
                length=10, lw=1)
    ax1.set_xlim(-(Axis_max-X*d)/2,(Axis_max+X*d)/2)
    ax1.set_ylim(-(Axis_max-Y*d)/2,(Axis_max+Y*d)/2)
    ax1.set_zlim(-(Axis_max-Z*d)/2,(Axis_max+Z*d)/2)
    ax1.set_xlabel("x[nm]")
    ax1.set_ylabel("y[nm]")
    ax1.set_zlabel("z[nm]")
    ax1.grid(False)
    fig1.suptitle(f"E field - Arrow plot\n {E_dir}")
    plt.savefig(os.path.join(position, "E-Field_Vector", f"E_field_arrow{iteration}.png"))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    geo_dic = set()
    surf_list = []
    surf_color = []
    x_grid, y_grid, z_grid = (np.indices((X+1,Y+1,Z+1))-0.5)*d
    filled, colors = np.zeros((X,Y,Z)), np.zeros((X,Y,Z))
    for i,pos in enumerate(geometry):
        geo_dic.add((pos[0],pos[1],pos[2]))
    for i,pos in enumerate(geometry):
        if (pos[0]+1,pos[1],pos[2]) not in geo_dic or (pos[0]-1,pos[1],pos[2]) not in geo_dic or\
           (pos[0],pos[1]+1,pos[2]) not in geo_dic or (pos[0],pos[1]-1,pos[2]) not in geo_dic or\
           (pos[0],pos[1],pos[2]+1) not in geo_dic or (pos[0],pos[1],pos[2]-1) not in geo_dic:
            filled[pos[0],pos[1],pos[2]] = 1
            colors[pos[0],pos[1],pos[2]] = E_tot_abs[i]
    surf_list = np.array(surf_list)
    surf_color = np.array(surf_color)

    colors = cm.Spectral_r(norm(colors))
    ax2.voxels(x_grid,y_grid,z_grid,filled.astype(bool),facecolors=colors,linewidth=0.5)

    ax2.set_xlim(-(Axis_max-X*d)/2,(Axis_max+X*d)/2)
    ax2.set_ylim(-(Axis_max-Y*d)/2,(Axis_max+Y*d)/2)
    ax2.set_zlim(-(Axis_max-Z*d)/2,(Axis_max+Z*d)/2)
    ax2.set_xlabel("x[nm]")
    ax2.set_ylabel("y[nm]")
    ax2.set_zlabel("z[nm]")
    ax2.grid(False)
    fig2.colorbar(colorset, shrink=0.9, aspect=10, cax=ax2.inset_axes([-0.1, 0.1, 0.05, 0.8]))

    fig2.suptitle(f"E field - Scatter plot\n, {k_dir} - iteration{iteration}") 
    plt.savefig(os.path.join(position, "E-Field_Vector", f"E_field{iteration}.png"))
