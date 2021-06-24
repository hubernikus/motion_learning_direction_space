#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import os

import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles
from dynamic_obstacle_avoidance.visualization.gamma_field_visualization import gamma_field_multihull

# from motion_learning_direction_space.learner.directional import DirectionalGMM
from motion_learning_direction_space.learner.directional_gmm import DirectionalGMM
from motion_learning_direction_space.graph import GraphGMM
# motion_learning_direction_space/visualization/
from motion_learning_direction_space.visualization.convergence_direction import test_convergence_direction_multihull


if (__name__) == "__main__":
    plt.close('all')
    plt.ion() # continue program when showing figures
    save_figure = True
    showing_figures = True
    name = None
    
    print("Start script .... \n")

    if True:
        name = "2D_Ashape"
        n_gaussian = 6

    if name is not None:
        dataset_name = os.path.join("dataset", name+".mat")

    if True: # relearn (debugging only)
        import numpy as np
        np.random.seed(4)
        MainLearner = GraphGMM(file_name=dataset_name, n_gaussian=n_gaussian)
        # MainLearner = DirectionalGMM()
        # MainLearner.load_data_from_mat(file_name=dataset_name)
        # MainLearner.regress(n_gaussian=n_gaussian)
        # gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)

    # MainLearner.plot_position_data()
    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)
    MainLearner.create_graph_from_gaussians()
    MainLearner.create_learned_boundary()

    
    if False:
        MainLearner.plot_obstacle_wall_environment()
        # plt.savefig(os.path.join("figures", name+"_convergence_direction" + ".png"), bbox_inches="tight")
                    
    pos_attractor = MainLearner.pos_attractor

    MainLearner.set_convergence_direction(attractor_position=pos_attractor)
    x_lim, y_lim = MainLearner.get_xy_lim_plot()

    plot_gamma_value = False
    if plot_gamma_value:
        n_subplots = 6
        n_cols = 3
        n_rows = int(n_subplots / n_cols)
        # fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 5))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

        for it_obs in range(n_subplots):
            it_x = it_obs % n_rows
            it_y = int(it_obs / n_rows)
            ax = axs[it_x, it_y]
            
            gamma_field_multihull(MainLearner, it_obs,
                                  n_resolution=100, x_lim=x_lim, y_lim=y_lim, ax=ax)

            # test_convergence_direction_multihull(
                # MainLearner, it_obs, n_resolution=30, x_lim=x_lim, y_lim=y_lim, ax=ax)
            
        plt.subplots_adjust(wspace=0.001, hspace=0.001)
        if save_figure:
            plt.savefig(os.path.join("figures", "gamma_value_subplots" + ".png"),
                        bbox_inches="tight")
            # plt.savefig(os.path.join("figures", "test_convergence_direction" + ".png"),
                        # bbox_inches="tight")

    plot_vectorfield = True
    if plot_vectorfield:
        n_resolution = 30
        
        def initial_ds(position):
            return evaluate_linear_dynamical_system(position, center_position=pos_attractor)
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        Simulation_vectorFields(
            x_lim, y_lim, n_resolution,
            # obs=obstacle_list,
            obs=MainLearner,
            saveFigure=True,
            figName=name+"_converging_linear_base",
            noTicks=True, showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_ds,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            pos_attractor=pos_attractor,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,       
            )
        MainLearner.reset_relative_references()

        def initial_ds(position):
            # xd = self.predict(x_traj[:, 0, ii])
            return MainLearner.predict(position)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        Simulation_vectorFields(
            x_lim, y_lim, n_resolution,
            # obs=obstacle_list,
            obs=MainLearner,
            saveFigure=True,
            figName=name+"_converging_nonlinear_base",
            noTicks=True, showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_ds,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            pos_attractor=pos_attractor,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,       
            )
        MainLearner.reset_relative_references()


        # plt.savefig(os.path.join("figures", name+"_converging_linear_base" + ".png"), bbox_inches="tight")
    plt.show()
    
print("\n\n\n... script finished.")


