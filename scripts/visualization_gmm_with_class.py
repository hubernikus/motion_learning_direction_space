#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ =  "lukashuber"
__date__ = "2021-05-16"
# __date__ = "2021-05-16"

import matplotlib.pyplot as plt

from motion_learning_direction_space.learner.directional import DirectionalLearner


if (__name__) == "__main__":
    plt.ion() # continue program when showing figures

    saveFigure = False
    save_dir = "../figures/"
    save_directory = "figures"
    showing_figures = True

    # plt.close("all")
    print("Start script .... \n")

    # dataset_name = "dataset/2D_messy-snake.mat"
    # n_gaussian = 17

    dataset_name = "dataset/2D_incremental_1.mat"
    n_gaussian = 5

    # dataset_name = "dataset/2D_Sshape.mat"
    # n_gaussian = 5

    # dataset_name = "dataset/2D_Ashape.mat"
    # n_Gaussian = 6

    MainLearner = DirectionalLearner()
    MainLearner.load_data_from_mat(file_name=dataset_name)
    MainLearner.regress(n_gaussian=n_gaussian)
    # MainLearner.plot_position_and_gaussians_2d()
    # MainLearner.plot_time_direction_and_gaussians()

    MainLearner.plot_data_and_gaussians(n_grid=30)
    # MainLearner.plot_vector_field_weights(n_grid=10)

    plt.show()
    
print("\n\n\n... script finished.")

