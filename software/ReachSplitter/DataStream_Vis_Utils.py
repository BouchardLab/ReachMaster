import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import seaborn as sns
import pandas
import scipy
import pdb

### MAKE MATLAB BEAUTIFUL
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
sns.set(font='Franklin Gothic Book',
        rc={
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False})
sns.set_context("notebook", rc={"font.size": 16,
                                "axes.titlesize": 20,
                                "axes.labelsize": 18})

plt.legend(frameon=False)


def get_single_trial(df, date, session, rat):
    rr = df.loc[df['Date'] == date]
    rr = rr.loc[rr['S'] == session]
    new_df = rr.loc[rr['rat'] == rat]
    return new_df


def sample_around_point(list_of_data, n):
    l = []
    for i in list_of_data:
        d = sample_around_n_rand(i, n)
        l.append(d)
    l = np.asarray(l)
    return l


def sample_around_n_rand(i, n):
    d = np.random.uniform(i - .65, i + .65, size=(n, 1))
    d = np.random.permutation((d))
    return d


def make_list_pts(x_array, y_array, z_array):
    big_x = []
    big_y = []
    big_z = []
    for c, v in enumerate(x_array):
        cx = sample_around_point()
    return cx


def k3d_plot(x, y, z):
    x_rewz = [4.2, 1.5, 1.5, 4.3]
    y_rewz = [-4.5, 2.03, 2.03, -4.5]
    z_rewz = [3.3, 10.4, -10.6, -13.1]
    p = np.vstack([x, y, z]).T
    indices = Triangulation(x_rewz, y_rewz).triangles.astype(np.uint32)
    plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=p, point_size=0.2)
    plot += plt_points
    plt_mesh = k3d.mesh(np.vstack([x_rewz, y_rewz, z_rewz]).T, indices,
                        color_map=k3d.colormaps.basic_color_maps.Jet,
                        attribute=z,
                        color_range=[-1.1, 2.01])
    plot += plt_mesh
    plt_points.shader = '3d'
    plot.display()


def oned_plot(X, Y, Z, zeros, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    elev = -180
    azim = -90
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, zeros, zeros, marker='o', color='r', s=6, label='Reach Locations 1-D (X)')
    ax.scatter(zeros, Y, zeros, marker='o', color='g', s=6, label='Reach Locations 1-D (Y)')
    ax.scatter(zeros, zeros, Z, marker='o', color='b', s=6, label='Reach Locations 1-D (Z)')
    ax.scatter(0, 0, 0, marker='x', color='k', s=40, label='Origin')
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='m', s=20, label='Handle Initialization Positions')
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='k', label='Reward Zone')
    offset = [20, 20, 15]
    ax.quiver(20, 10, 15, 1, 0, 0, length=8, linewidths=5, color='r', alpha=0.8)
    ax.text(25, 20, 10, '%s' % ('X'), size=20, zorder=1,
            color='r')
    ax.quiver(20, 10, 15, 0, 1, 0, length=8, linewidths=5, color='g', alpha=0.8)
    ax.text(23, 15, 18, '%s' % ('Y'), size=20, zorder=1,
            color='g')
    ax.quiver(20, 10, 15, 0, 0, 1, length=8, linewidths=5, color='b', alpha=0.8)
    ax.text(24, -5, 30, '%s' % ('Z'), size=20, zorder=1,
            color='b')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    plt.legend()
    # ax.view_init(elev,azim)
    plt.title('Reaching Workspace for Fall 2020 Experiments- 3D Points: 1-D Line(s)')
    if savepath:
        plt.savefig('Reaching_1dplanes_FinalF2020.png')
    plt.show()


def single_plot(X, Y, Z, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    elev = -180
    azim = -90
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, Y, Z, marker='o', color='b', s=6, label='Reach Locations 2-D (X-Y Plane)')
    # ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z),color='y')
    ax.scatter(0, 0, 0, marker='x', color='black', s=40, label='Origin')
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='r', s=20, label='Starting Positions')
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='r', label='Reward Zone')
    XN, YN = np.meshgrid(np.ravel(X), np.ravel(Y))
    ZN = np.ravel(Z)
    ax.plot_surface(XN, YN, ZN.reshape(ZN.shape[0], 1), rstride=2, cstride=2, color='b', alpha=0.4, linewidth=0.0001,
                    edgecolors='g')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    plt.legend()
    plt.title('Reaching Workspace for Fall 2020 Experiments- 3D Points')
    if savepath:
        plt.savefig('Reaching_1dplanes_xy.png')
    plt.show()


def twod_plot(X, Y, Z, zeros, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False, surfaces=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    elev = -180
    azim = -90
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, Y, Z, marker='o', color='r', s=6, label='Reach Locations 2-D (X-Y)')
    # ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z),color='b',label='X-Y')
    ax.scatter(X, Z, Y, marker='o', color='g', s=6, label='Reach Locations 2-D (X-Z)')
    # ax.plot(np.ravel(X),np.ravel(Z),np.ravel(Y),color='m',label='X-Z')
    ax.scatter(Z, X, Y, marker='o', color='b', s=6, label='Reach Locations 2-D (Y-Z)')
    # ax.plot(np.ravel(Z),np.ravel(X),np.ravel(Y),color='y',label='Y-Z')
    ax.scatter(0, 0, 0, marker='x', color='k', s=40, label='Origin')
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='m', s=20, label='Origin')
    # take Reward Zone coordinates from config file, use forward kinematics to transform
    # these values are hard-coded
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='k', label='Reward Zone')
    # Making Surfaces: Re-sizing/MatPlotlib
    if surfaces:
        XN, YN = np.meshgrid(np.ravel(X), np.ravel(Y))
        ZN = np.ravel(Z)
        ax.plot_surface(XN, YN, ZN.reshape(ZN.shape[0], 1), rstride=1, cstride=1, color='r', alpha=0.2, linewidth=0,
                        edgecolors='r', antialiased=True)
        XN1, YN1 = np.meshgrid(np.ravel(X), np.ravel(Z))
        ZN1 = np.ravel(Y)
        ax.plot_surface(XN1, YN1, ZN1.reshape(ZN1.shape[0], 1), rstride=1, cstride=1, color='g', alpha=0.2, linewidth=0,
                        edgecolors='g', antialiased=True)
        XN2, YN2 = np.meshgrid(np.ravel(Z), np.ravel(X))
        ZN2 = np.ravel(Y)
        ax.plot_surface(XN2, YN2, ZN2.reshape(ZN2.shape[0], 1), rstride=1, cstride=1, color='b', alpha=0.2, linewidth=0,
                        edgecolors='b', antialiased=True)
    ax.quiver(20, 10, 15, 1, 0, 0, length=8, linewidths=5, color='r', alpha=0.8)
    ax.text(25, 20, 10, '%s' % ('X'), size=20, zorder=1,
            color='r')
    ax.quiver(20, 10, 15, 0, 1, 0, length=8, linewidths=5, color='g', alpha=0.8)
    ax.text(23, 15, 18, '%s' % ('Y'), size=20, zorder=1,
            color='g')
    ax.quiver(20, 10, 15, 0, 0, 1, length=8, linewidths=5, color='b', alpha=0.8)
    ax.text(24, -5, 30, '%s' % ('Z'), size=20, zorder=1,
            color='b')
    # ax.quiver(xi,yi,zi,ui,vi,wi,arrow_length_ratio=1)
    ax.set_xlabel('x (pos mm)')
    ax.set_ylabel('y (pos mm)')
    ax.set_zlabel('z (pos mm)')
    plt.legend()
    plt.title('Reaching Workspace for Fall 2020 Experiments- 2D Planes (X-Y,Y-Z, X-Z)')
    if savepath:
        plt.savefig('Plane_Scatter_Points.png')
    plt.show()
    return


def xform_coords_euclidean(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z


def inverse_xform_coords(r, theta_y, theta_z):
    xgimbal_xoffset = 168
    ygimbal_to_joint = 64
    ygimbal_yoffset = 100
    zgimbal_to_joint = 47
    zgimbal_zoffset = 117
    x_origin = 1024
    y_origin = 608
    z_origin = 531
    Ax = np.sqrt(
        xgimbal_xoffset ** 2 + r ** 2 - 2 * xgimbal_xoffset * r * np.cos(theta_y) * np.cos(theta_z)
    )
    gammay = -np.arcsin(
        np.sin(theta_y) *
        np.sqrt(
            (r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        ) /
        np.sqrt(
            (xgimbal_xoffset - r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        )
    )
    gammaz = -np.arcsin(r * np.sin(theta_z) / Ax)
    Ay = np.sqrt(
        (ygimbal_to_joint - ygimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_yoffset - ygimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Az = np.sqrt(
        (zgimbal_to_joint - zgimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_zoffset - zgimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Ax = np.round((Ax - xgimbal_xoffset) / 50 * 1024 + x_origin, decimals=1)  # bits
    Ay = np.round((Ay - ygimbal_yoffset) / 50 * 1024 + y_origin, decimals=1)  # bits
    Az = np.round((Az - zgimbal_zoffset) / 50 * 1024 + z_origin, decimals=1)  # bits
    # convert tranformed commands to appropriate data types/format

    return Ax, Ay, Az


def forward_xform_coords(x, y, z):
    Axx = 168
    Ly = 64
    Ayy = 100
    Lz = 47
    Azz = 117
    X0 = 1024
    Y0 = 608
    Z0 = 531
    Ax_est = (x - X0) / (1024 * 50) + Axx
    Ay_est = (y - Y0) / (1024 * 50) + Ayy
    Az_est = (z - Z0) / (1024 * 50) + Azz
    c1 = np.asarray((0, 0, 0))
    c2 = np.asarray((Ly, Ayy, 0))
    c3 = np.asarray((Lz, 0, Azz))
    u = np.asarray((Ly, Ayy, 0)) / np.sqrt(Ly ** 2 + Ayy ** 2)
    v = c3 - np.dot(c3, u) * u
    v = v / np.sqrt(np.dot(v, v))
    w = np.cross(u, v)
    y1 = np.asarray((0, 1, 0))
    z1 = np.asarray((0, 0, 1))
    U2 = np.sqrt(np.sum((c2 - c1) ** 2))
    U3 = np.dot(c3, u)
    V3 = np.dot(c3, v)
    sd = np.dot(c3, c3)
    cos_top = (Az_est ** 2 + Lz ** 2 - sd)
    cos_bot = (2 * Az_est * Lz)
    r3 = np.sqrt(
        Az_est ** 2 + (Ly - Lz) ** 2 - (2 * Az_est * (Ly - Lz) * np.cos(np.pi - np.arccos((Az_est ** 2 + Lz ** 2 - sd)
                                                                                          / (2 * Az_est * Lz)))))
    Pu = (Ly ** 2 - Ay_est ** 2 + U2 ** 2) / (2 * U2)
    Pv = (U3 ** 2 + V3 ** 2 - 2 * U3 * Pu + Ly ** 2 - r3 ** 2) / (2 * V3)
    Pw = np.sqrt(-Pu ** 2 - Pv ** 2 + Ly ** 2)
    Py = Pu * np.dot(u, y1) + Pv * np.dot(v, y1) + Pw * np.dot(w, y1)
    Pz = Pu * np.dot(u, z1) + Pv * np.dot(v, z1) + Pw * np.dot(w, z1)
    gammay_est = np.arcsin(Py / (Ly * np.cos(np.arcsin(Pz / Ly))))
    gammaz_est = np.arcsin(Pz / Ly)
    r = np.sqrt(Axx ** 2 + Ax_est ** 2 - (2 * Axx * Ax_est * np.cos(gammay_est) * np.cos(gammaz_est)))
    dz = np.sin(-gammaz_est)
    dy = np.sin(-gammay_est)
    theta = np.arcsin(dz * Ax_est / r)
    phi = np.arcsin(Ax_est * dy * np.cos(-gammaz_est) / r / np.cos(theta))
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    return r, theta, phi, x, y, z


def euclidean_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    theta = np.arcsin(y / (r * np.sin(phi)))
    return r, theta, phi
