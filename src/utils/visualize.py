import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import pandas as pd

# from utils import simtacls_utils
# from simtacls import config

# tactile_skin = simtacls_utils.TactileSkin(skin_path='./resources/skin.vtk')
# # extract fixed inward radial vectors
# radial_vectors = tactile_skin.get_radial_vectors()

class TactileVisualize():
    def __init__(self, skin_path,
                       init_positions, 
                       depth_range=[-5, 16]):
        self.skin_path = skin_path
        self.init_positions = init_positions
        self.deviations = np.zeros_like(self.init_positions) # running deviations
        self.depth_range =  depth_range
        self.plot_initialize()


    def plot_initialize(self):
        # set camera pososition
        self.cpos = np.array([[-57.34826013561122, -661.242740117575, 352.06298028880514],
                              [-0.056080635365844955, -1.209135079294247, 118.18100834810429],
                              [0.024125960714330284, 0.3320410642889663, 0.9429563455672065]])
        
        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.set_background("white", top="white")
        pv.global_theme.font.color = 'black' 
        pv.global_theme.font.title_size = 16 
        pv.global_theme.font.label_size = 16  
        boring_cmap = plt.cm.get_cmap("bwr")  
        
        self.plotter.subplot(0, 0)
        self.plotter.camera_position = self.cpos
        self.plotter.show_axes()
        self.skin_est = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.deviations, axis=1)
        self.skin_est['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.plotter.add_mesh(self.skin_est, cmap=boring_cmap, clim=self.depth_range)

    def updade_plot(self, deformation, positions):
        self.skin_est['contact depth (unit:mm)'] = deformation # for contact depth visualization
        self.skin_est.points = positions


class TacThermalSkinVisualize():
    def __init__(self, skin_path, 
                       init_positions,
                       depth_display_range,
                       plot_init = True):
        self.skin_path = skin_path
        self.positions = np.array(init_positions) # running positions
        self.deviations = np.zeros_like(self.positions) # running deviations
        self.depth_display_range = depth_display_range

        # set camera pososition
        self.cpos = np.array([[-57.34826013561122, -661.242740117575, 352.06298028880514],
                              [-0.056080635365844955, -1.209135079294247, 118.18100834810429],
                              [0.024125960714330284, 0.3320410642889663, 0.9429563455672065]])
        
        # update only once as initialization
        if plot_init:
            self.plot_initialize()

    def plot_initialize(self):
        # self.plotter = pvqt.BackgroundPlotter(shape=(1, 2))
        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.set_background("white", top="white")
        pv.global_theme.font.color = 'black' 
        pv.global_theme.font.title_size = 16 
        pv.global_theme.font.label_size = 16  
        boring_cmap = plt.cm.get_cmap("cool")  
        
        # self.plotter.subplot(0, 0)
        self.plotter.camera_position = self.cpos
        self.plotter.show_axes()
        self.skin_est = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.deviations, axis=1)
        self.skin_est['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin_est.points = self.positions
        self.plotter.add_mesh(self.skin_est, 
                              cmap=boring_cmap, 
                              clim=self.depth_display_range,
                              show_edges=True)

        # self.plotter.subplot(0, 1)
        # self.plotter.camera_position = self.cpos
        # self.plotter.show_axes()
        # self.skin_gt = pv.read(self.skin_path) # for pyvista visualization
        # norm_deviations = np.linalg.norm(self.deviations, axis=1)
        # self.skin_gt['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        # self.skin_gt.points = self.positions
        # self.plotter.add_mesh(self.skin_gt, cmap=boring_cmap, clim=self.depth_display_range)


class BarrelSkinVisualize():
    def __init__(self, skin_path, 
                       init_positions, 
                       plot_init = True):
        self.skin_path = skin_path
        self.init_positions = init_positions
        self.positions = np.array(init_positions)
        if plot_init:
            self.plot_initialize()

    def plot_initialize(self):
        self.plotter = pvqt.BackgroundPlotter(shape=(1, 2))
        boring_cmap = plt.cm.get_cmap("bwr")       
        self.skin = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.positions - self.init_positions, axis=1)
        
        axes = pv.Axes(show_actor=True, actor_scale = 2.0, line_width=5)

        self.plotter.subplot(0, 0)
        self.plotter.show_axes()
        # self.plotter.camera.position = (300, 0, 400)
        # # self.plotter.camera.view_angle = 60.0
        # self.plotter.camera.roll = 135.0
        self.skin['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin.points = self.positions
        self.skin.rotate_x(270, point=axes.origin)
        self.plotter.add_mesh(self.skin, cmap=boring_cmap, clim=[0, 13])

        self.plotter.subplot(0, 1)
        self.plotter.show_axes()
        self.plotter.camera.position = (45, 300, 52)
        self.plotter.camera.roll = -145.0
        self.skin['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin.points = self.positions
        self.plotter.add_mesh(self.skin, cmap=boring_cmap, clim=[0, 13])


class MassageArmVisualize():
    def __init__(self, skin_path, init_positions, plot_init = True):
        self.skin_path = skin_path
        self.init_positions = init_positions
        self.positions = np.array(init_positions)
        if plot_init:
            self.plot_initialize()

    def plot_initialize(self):
        self.plotter = pvqt.BackgroundPlotter(shape=(1, 2))
        boring_cmap = plt.cm.get_cmap("bwr")       
        self.skin = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.positions - self.init_positions, axis=1)
        
        self.plotter.subplot(0, 0)
        self.skin['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin.points = self.positions
        self.plotter.add_mesh(self.skin, cmap=boring_cmap, clim=[0, 6])

        self.plotter.subplot(0, 1)
        self.skin['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin.points = self.positions
        self.plotter.add_mesh(self.skin, cmap=boring_cmap, clim=[0, 6])

def get_file_idx(node_idx_path, label_idx_path):
    df_node_idx = pd.read_csv(node_idx_path)
    df_label_idx = pd.read_csv(label_idx_path)

    node_idx = np.array(df_node_idx.iloc[:,0], dtype=int) # (full skin) face node indices in vtk file exported from SOFA 
    node_idx = list(set(node_idx)) # eleminate duplicate elements (indices)
    node_idx = sorted(node_idx) # sorted the list of indices

    label_idx = list(df_label_idx.iloc[:,0]) #(not full skin) at nodes used for training - labels
    file_idx = [node_idx.index(idx) for idx in label_idx]

    return file_idx


def points_extraction(vtk_path):
    """Extract 3D-coordinated points in VTK files
    for initial/non-deformed points:

    Parameters:
        vtk_path (file with .vtk extension) --- sensing skin represented by mesh in VTK format
    Returns:
        nodes of mesh
    """
    points_ls = []
    with open(vtk_path, 'r') as rf:
        for idx, line in enumerate(rf):
            if 6 <= idx <= 712:
                points = [float(x) for x in line.split()]
                points_ls.append(points)
            elif idx > 712:
                break

    return np.array(points_ls, dtype=float)
