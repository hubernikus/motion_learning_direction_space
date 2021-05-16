#!/usr/bin/python3
'''
Tools to simplify visualization

'''
__author__ =  "lukashuber"
__date__ = "2021-05-16"

import sys
import warnings
import random

import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion() # continue program when showing figures

# from math import pi

# import scipy.io # import *.mat files -- MATLAB files

# Machine learning datasets
# from sklearn.mixture import GaussianMixture
# from sklearn.mixture import BayesianGaussianMixture
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from sklearn import mixture

colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red', 'green', 'purple', 'black', 'violet', 'tan']
# colors = ['navy']
def draw_gaussians(gmm, ax, plot_dims):
    # for n, color in enumerate(colors):
    n_gaussian = gmm.n_components
    for n in range(n_gaussian):
        color = colors[np.mod(n,len(colors))]
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][plot_dims,:][:,plot_dims]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[plot_dims,:][:,plot_dims]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][plot_dims])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        # ell = mpl.patches.Ellipse(gmm.means_[n, plot_dims], v[0], v[1], 180 + angle, color=color)
        ell = mpl.patches.Ellipse(gmm.means_[n, plot_dims], v[0], v[1], 180 + angle, color=color)
                                  
        
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.plot(gmm.means_[n, plot_dims[0]], gmm.means_[n, plot_dims[1]], 'k.', markersize=12, linewidth=30)
        # ax.plot(gmm.means_[n, 0], gmm.means_[n, 1], 'k+', s=12)
