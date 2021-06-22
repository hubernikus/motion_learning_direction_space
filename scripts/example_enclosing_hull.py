#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""
# Author: Lukas Huber
# Date: 2021-05-16
# License: BSD (c) 2021

import os

import matplotlib.pyplot as plt

# from motion_learning_direction_space.learner.directional import DirectionalGMM
from motion_learning_direction_space.learner.directional_gmm import DirectionalGMM
from motion_learning_direction_space.graph import GraphGMM

if (__name__) == "__main__":
    plt.close('all')
    plt.ion() # continue program when showing figures
    save_figure = False
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
        np.random.seed(0)
        MainLearner = GraphGMM()
        # MainLearner = DirectionalGMM()
        MainLearner.load_data_from_mat(file_name=dataset_name)
        MainLearner.regress(n_gaussian=n_gaussian)

        gauss_colors = MainLearner.complementary_color_picker(n_colors=n_gaussian)

    # MainLearner.plot_position_data()
    # MainLearner.plot_position_and_gaussians_2d(colors=gauss_colors)
    MainLearner.create_graph()
    # MainLearner.plot_graph_and_gaussians()
    
    MainLearner.create_learned_boundary()
    MainLearner.plot_obstacle_wall_environment()

    
    
    plt.show()
    
print("\n\n\n... script finished.")


