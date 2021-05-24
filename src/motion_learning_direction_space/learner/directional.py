#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ =  "lukashuber"
__date__ = "2021-05-16"


import sys
import os
import warnings

from functools import lru_cache

from math import pi
import numpy as np

import scipy.io # import *.mat files -- MATLAB files

# Machine learning datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import mixture

# Quadratic programming
# from cvxopt.solvers import coneqp

# Custom libraries
# from motion_learning_direction_space.visualization.gmm_visualization import draw_gaussians
from motion_learning_direction_space.math_tools import rk4, mag_linear_maximum
# from motion_learning_direction_space.direction_space import velocity_reduction, regress_gmm, get_gaussianProbability, velocity_reconstruction, velocity_reduction, get_mixing_weights, get_mean_yx
from motion_learning_direction_space.learner.base import Learner
from motion_learning_direction_space.learner.visualizer import LearnerVisualizer

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

class DirectionalLearner(LearnerVisualizer, Learner):
    """ Virtual class to learn from demonstration / implementation. """
    def __init__(self, directory_name='', file_name=''):
        self.directory_name = directory_name
        self.file_name = file_name

        # where should this ideally be defined?
        self.dim = 2

        self.pos = None
        self.vel = None

        # Noramlized etc. regression data
        self.X = None

    @property
    def num_gaussians(self):
        return self.dpgmm.covariances_.shape[0]
        
    def load_data_from_mat(self, file_name=None, dims_input=None):
        """ Load data from file mat-file & evaluate specific parameters """
        if file_name is not None:
            self.file_name = file_name

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name))

        if dims_input is None:
            self.dims_input = [0,1]
        
        ii = 0 # Only take the first fold.
        self.pos = self.dataset['data'][0, ii][:2, :].T
        self.vel = self.dataset['data'][0, ii][2:4, :].T
        t = np.linspace(0, 1, self.dataset['data'][0, ii].shape[1])
        
        pos_attractor =  np.zeros((self.dim))
        
        for it_set in range(1, self.dataset['data'].shape[1]):
            self.pos = np.vstack((self.pos, self.dataset['data'][0,it_set][:2,:].T))
            self.vel = np.vstack((self.vel, self.dataset['data'][0,it_set][2:4,:].T))

            pos_attractor = (pos_attractor
                             + self.dataset['data'][0,it_set][:2,-1].T
                             / self.dataset['data'].shape[1])
                             
        
            # TODO include velocity - rectify
            t = np.hstack((t, np.linspace(0, 1, self.dataset['data'][0,it_set].shape[1])))

        direction = get_angle_space_of_array(
            directions=self.vel.T, positions=self.pos.T,
            func_vel_default=evaluate_linear_dynamical_system)
    
        self.X = np.hstack((self.pos, direction.T, np.tile(t, (1, 1)).T))

        self.num_samples = self.X.shape[0]

        weightDir = 4
        # Normalize dataset
        self.meanX = np.mean(self.X, axis=0)

        self.meanX = np.zeros(4)
        # X = X - np.tile(meanX , (X.shape[0],1))
        self.varX = np.var(self.X, axis=0)

        # All distances should have same variance
        self.varX[:self.dim] = np.mean(self.varX[:self.dim])
        
        # All directions should have same variance
        self.varX[self.dim:2*self.dim-1] = np.mean(self.varX[self.dim:2*self.dim-1])
        
        # Stronger weight on directions!
        self.varX[self.dim:2*self.dim-1] = self.varX[self.dim:2*self.dim-1]*1/weightDir 

        self.X = self.X / np.tile(self.varX, (self.X.shape[0], 1))
        self.pos_attractor = (pos_attractor-self.meanX[:self.dim]) / self.varX[:self.dim]

    def regress(self, n_gaussian=5, tt_ratio=0.75):
        """ Regress based on the data given."""
        a_label = np.zeros(self.num_samples)
        all_index = np.arange(self.num_samples)
        
        train_index, test_index = train_test_split(all_index, test_size=(1-tt_ratio))
        
        X_train = self.X[train_index, :]
        X_test = self.X[test_index, :]
        
        y_train = a_label[train_index]
        y_test = a_label[test_index]

        cov_type = 'full'

        self.dpgmm = mixture.BayesianGaussianMixture(
            n_components=n_gaussian, covariance_type='full')

        # sample dataset
        reference_dataset = 0
        n_start = 0
        for it_set in range(reference_dataset):
            n_start +=  dataset['data'][0,it_set].shape[1]
            
        index_sample = [int(np.round(n_start + self.dataset['data'][0, reference_dataset].shape[1]
                                     /n_gaussian*ii)) for ii in range(n_gaussian)]

        self.dpgmm.means_init = self.X[index_sample, :]
        self.dpgmm.means_init = X_train[np.random.choice(np.arange(n_gaussian)), :]

        self.dpgmm.fit(X_train[:, :])

    def predict(self, xx):
        """ Predict learned DS based on Dynamical system evaluation."""
        output_gmm  = self.regress_gmm(np.array([xx]))
        dir_angle_space = np.squeeze(output_gmm)[:self.dim-1]

        null_direction = evaluate_linear_dynamical_system(xx)
        vel = get_angle_space_inverse(dir_angle_space=dir_angle_space,
                                      null_direction=null_direction)
        return np.squeeze(vel)

    # @lru_cache(maxsize=5)
    def predict_array(self, xx):
        """ Predict based on learn model and xx being an input matrix
        with multiple datapoints. """
        output_gmm = self.regress_gmm(xx.T)
        vel = get_angle_space_inverse_of_array(
            vecs_angle_space=output_gmm[:, :self.dim-1].T,
            positions=xx, func_vel_default=evaluate_linear_dynamical_system)
        
        return vel

    def transform_initial_to_normalized(self, values, dims_ind):
        """ Inverse-normalization and return the modified value. """
        n_samples = values.shape[0]
        if self.meanX is not None:
           values = (values - np.tile(self.meanX[dims_ind], (n_samples,1)) )
        if self.varX is not None:
           values = values/np.tile(self.varX[dims_ind], (n_samples,1))
        return values

    def transform_normalized_to_initial(self, values, dims_ind):
        """ Inverse-normalization and return the modified values. """
        n_samples = values.shape[0]
        
        if self.varX is not None:
           values = values*np.tile(self.varX[dims_ind], (n_samples,1))
           
        if self.meanX is not None:
           values = (values+np.tile(self.meanX[dims_ind], (n_samples,1)) )

        return values

    def regress_gmm(self, X, input_output_normalization=True,
                    convergence_attractor=True, p_beta=2, beta_min=0.5, beta_r=0.3):
        """ Evaluate the regress field at all the points X""" 
        # output_gmm = self.regress_gmm(pos_x.T, self.dpgmm, self.self.dims_input,
                                      # self.meanX, self.varX, attractor=self.pos_attractor)
        dim = self.dpgmm.covariances_[0].shape[1]
        dim_in = np.array(self.dims_input).shape[0]
        n_samples = X.shape[0]
        n_gaussian = self.dpgmm.covariances_.shape[0]

        dims_output = [gg for gg in range(dim) if gg not in self.dims_input]

        if input_output_normalization:
            X = self.transform_initial_to_normalized(X, dims_ind=self.dims_input)

        beta = self.get_mixing_weights(X)
        mu_yx = self.get_mean_yx(X)

        if convergence_attractor:
            if self.pos_attractor is not None: # zero attractor
            # dist_attr = np.linalg.norm(X-np.tile(attractor, (n_samples,1)) , axis=1)
                dist_attr = np.linalg.norm(X - np.tile(self.pos_attractor, (n_samples, 1)) , axis=1)
            else:
                dist_attr = np.linalg.norm(X, axis=1)

            beta = np.vstack((beta, np.zeros(n_samples)))

            # Zero values
            beta[:,dist_attr==0] = 0
            beta[-1,dist_attr==0] = 1

            # Nonzeros values
            beta[-1,dist_attr!=0] =  (dist_attr[dist_attr!=0]/beta_r)**(-p_beta) + beta_min 
            beta[:,dist_attr!=0] = (beta[:, dist_attr!=0] / np.tile(
                np.linalg.norm(beta[:,dist_attr!=0], axis=0), (self.num_gaussians+1, 1)))

            mu_yx = np.dstack((mu_yx, np.zeros((dim-dim_in, n_samples,1))))

        regression_value = np.sum( np.tile(beta.T, (dim-dim_in, 1, 1) ) * mu_yx, axis=2).T

        if input_output_normalization:
            regression_value = self.transform_normalized_to_initial(
                regression_value, dims_ind=dims_output)
        
        return regression_value

    # @lru_cache(maxsize=5)
    def get_mixing_weights(self, X, input_needs_normalization=False,
                           normalize_probability=False, weight_fac=5):
        """ Get input positions X of the form [dimension, number of samples]. """
        if input_needs_normalization:
            X = self.transform_initial_to_normalized(X, self.dims_input)
            
        n_samples = X.shape[0]
        n_gaussian = self.dpgmm.covariances_.shape[0]

        prob_gaussian = self.get_gaussian_probability(X)
        sum_probGaussian = np.sum(prob_gaussian, axis=0)

        alpha_times_prob = np.tile(self.dpgmm.weights_, (n_samples, 1)).T  * prob_gaussian

        if normalize_probability:
            beta = alpha_times_prob / np.tile(np.sum(alpha_times_prob, axis=0),
                                              (self.num_gaussians, 1))
        else:
            beta = alpha_times_prob
            # *weight_fac
            
        return beta

    def get_gaussian_probability(self, X):
        n_samples = X.shape[0]

        if not np.array(self.dims_input).shape[0]:
            self.dims_input = np.arange(self.dim)

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((self.num_gaussians, n_samples))

        for gg in range(self.num_gaussians):
            # Create function of this
            cov_matrix = self.dpgmm.covariances_[gg, :, :][self.dims_input, :][:, self.dims_input]
            fac = 1/((2*pi)**(self.dim*.5)*(np.linalg.det(cov_matrix))**(0.5))

            dX = X - np.tile(self.dpgmm.means_[gg, self.dims_input], (n_samples,1))

            val_pow = np.sum(np.tile(np.linalg.pinv(cov_matrix), (n_samples, 1, 1))
                             *np.swapaxes(np.tile(dX,  (self.dim, 1, 1)), 0, 1), axis=2)
            
            val_pow = np.exp(-np.sum(dX*val_pow, axis=1))
            prob_gauss[gg, :] = fac*val_pow
        return prob_gauss

    def get_mean_yx(self, X, stretch_input_values=False):
        dim = self.dpgmm.covariances_[0].shape[0]
        dim_in = np.array(self.dims_input).shape[0]

        n_samples = X.shape[0]
        dims_output = [gg for gg in range(dim) if gg not in self.dims_input]

        mu_yx = np.zeros((dim-dim_in, n_samples, self.num_gaussians))
        mu_yx_test = np.zeros((dim-dim_in, n_samples, self.num_gaussians))

        for gg in range(self.num_gaussians):
            covariance_inverse = np.linalg.pinv(
                self.dpgmm.covariances_[gg][self.dims_input, :][:,self.dims_input])

            covariance_output_input = (
                self.dpgmm.covariances_[gg][dims_output, :][:, self.dims_input])
            
            for nn in range(n_samples): # TODO #speed - batch process!!
                mu_yx[:, nn, gg] = (
                    self.dpgmm.means_[gg, dims_output]
                    + covariance_output_input
                    @ covariance_inverse
                    @ (X[nn, :] - self.dpgmm.means_[gg, self.dims_input]))

        return mu_yx

    def integrate_trajectory(self, num_steps=200, delta_t=0.02, nTraj=3, starting_points=None,
                             convergence_err=0.01):
        if starting_points is None:
            x_traj = np.zeros((self.dim, num_steps, nTraj))
            for ii in range(nTraj):
                x_traj[:, 0, ii] = self.dataset['data'][0, ii][:2, 0]
        else:
            nTraj = starting_points.shape[1]
            x_traj = np.zeros((self.dim, num_steps, nTraj))
    
        for ii in range(nTraj):
            for nn in range(1, num_steps):
                x_traj[:, nn, ii] = rk4(
                    delta_t, x_traj[:, nn-1, ii], self.predict)

                if np.linalg.norm(x_traj[:, nn, ii]) < convergence_err:
                    print(f"Converged after {nn} iterations.")
                    break

        return x_traj

