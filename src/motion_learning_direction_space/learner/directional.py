#!/usr/bin/python3
"""
Directional [SEDS] Learning
"""

__author__ =  "lukashuber"
__date__ = "2021-05-16"


import sys
import os
import warnings

from math import pi
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
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
from motion_learning_direction_space.visualization.gmm_visualization import draw_gaussians
from motion_learning_direction_space.math_tools import rk4, mag_linear_maximum
# from motion_learning_direction_space.direction_space import velocity_reduction, regress_gmm, get_gaussianProbability, velocity_reconstruction, velocity_reduction, get_mixing_weights, get_mean_yx
from motion_learning_direction_space.learner.base import Learner

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system


class DirectionalLearner(Learner):
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
        """ Dynamical system evaluation."""
        # output_gmm = self.regress_gmm(np.array([xx]), self.dpgmm, self.dims_input,
                                      # self.meanX, self.varX, attractor=self.pos_attractor)
        output_gmm = self.regress_gmm(np.array([xx]), self.dpgmm, self.dims_input,
                                      self.meanX, self.varX, attractor=self.pos_attractor)


        vel = get_angle_space_inverse_of_array(
            vecs_angle_space=output_gmm[:, :self.dim-1],
            positions=np.array([xx]).T, func_vel_default=evaluate_linear_dynamical_system)

        vel = vel * np.tile(np.linalg.norm(xx, axis=0), (self.dim, 1)) # Adapt magnitude
        return np.squeeze(vel)
    

    def regress_gmm(self, X, gmm, dims_input, mu=None, var=None, convergence_attractor=True,
                    attractor=None, p_beta=2, beta_min=0.5, beta_r=0.3):
        # output_gmm = self.regress_gmm(pos_x.T, self.dpgmm, self.self.dims_input,
                                      # self.meanX, self.varX, attractor=self.pos_attractor)


        # TODO: use class variables
        mu = self.meanX
        var = self.varX
        
        dim = self.dpgmm.covariances_[0].shape[1]
        dim_in = np.array(self.dims_input).shape[0]
        n_samples = X.shape[0]
        n_gaussian = self.dpgmm.covariances_.shape[0]

        dims_output = [gg for gg in range(dim) if gg not in self.dims_input]

        if mu is not None:
           X = (X-np.tile(mu[self.dims_input], (n_samples,1)) )

        if var is not None:
           X = X/np.tile(var[self.dims_input], (n_samples,1))

        beta = self.get_mixing_weights(X, gmm, self.dims_input)
        mu_yx = self.get_mean_yx(X, gmm, self.dims_input)

        if convergence_attractor:
            if attractor is not None: # zero attractor
            # dist_attr = np.linalg.norm(X-np.tile(attractor, (n_samples,1)) , axis=1)
                dist_attr = np.linalg.norm(X - np.tile(attractor, (n_samples, 1)) , axis=1)
            else:
                dist_attr = np.linalg.norm(X, axis=1)

            beta = np.vstack((beta, np.zeros(n_samples)))

            # Zero values
            beta[:,dist_attr==0] = 0
            beta[-1,dist_attr==0] = 1

            # Nonzeros values
            beta[-1,dist_attr!=0] =  (dist_attr[dist_attr!=0]/beta_r)**(-p_beta) + beta_min 
            beta[:,dist_attr!=0] = beta[:,dist_attr!=0]/np.tile(np.linalg.norm(beta[:,dist_attr!=0],axis=0), (n_gaussian+1,1))

            mu_yx = np.dstack((mu_yx, np.zeros((dim-dim_in, n_samples,1)) ))

        regression_value = np.sum( np.tile(beta.T, (dim-dim_in, 1, 1) ) * mu_yx, axis=2).T

        if np.array(var).shape[0]:
           regression_value = regression_value*np.tile(var[dims_output], (n_samples,1)) 
        if np.array(mu).shape[0]:
           regression_value = (regression_value+np.tile(mu[dims_output], (n_samples,1)) )

        return regression_value

    def get_mixing_weights(self, X, gmm, dims_input):
        dim = X.shape[1]
        n_samples = X.shape[0]
        n_gaussian = gmm.covariances_.shape[0]

        prob_gaussian = self.get_gaussian_probability(X, gmm, dims_input)

        sum_probGaussian = np.sum(prob_gaussian, axis=0)

        alpha_times_prob = np.tile(gmm.weights_, (n_samples, 1)).T  * prob_gaussian

        beta = alpha_times_prob / np.tile( np.sum(alpha_times_prob, axis=0), (n_gaussian, 1) )

        return beta

    def get_gaussian_probability(self, X, dpgmm, dims_input=[]):
        dim = X.shape[1]
        n_samples = X.shape[0]
        n_gaussian = dpgmm.covariances_.shape[0]

        if not np.array(dims_input).shape[0]:
            dims_input = np.arange(dim)

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((n_gaussian, n_samples))

        for gg in range(n_gaussian):
            # Create function of this
            cov_matrix = dpgmm.covariances_[gg,:,:][dims_input,:][:,dims_input]
            fac = 1/((2*pi)**(dim*.5)*(np.linalg.det(cov_matrix))**(0.5))

            dX = X-np.tile(dpgmm.means_[gg,dims_input], (n_samples,1) )

            pow = np.sum(np.tile(np.linalg.pinv(cov_matrix), (n_samples, 1, 1) )  *np.swapaxes(np.tile(dX,  (dim,1,1) ), 0,1), axis=2)
            pow = np.exp(-np.sum(dX *pow, axis=1))

            prob_gauss[gg, :] = fac*pow
        return prob_gauss

    def get_mean_yx(self, X, gmm, dims_input):
        n_gaussian = gmm.covariances_.shape[0]
        dim = gmm.covariances_[0].shape[0]
        dim_in = np.array(dims_input).shape[0]

        n_samples = X.shape[0]
        dims_output = [gg for gg in range(dim) if gg not in dims_input]

        mu_yx = np.zeros((dim-dim_in, n_samples, n_gaussian))
        mu_yx_test = np.zeros((dim-dim_in, n_samples, n_gaussian))

        for gg in range(n_gaussian):
            for nn in range(n_samples): # TODO #speed - batch process!!
                mu_yx[:, nn, gg] = gmm.means_[gg,dims_output] + gmm.covariances_[gg][dims_output,:][:,dims_input] @ np.linalg.pinv(gmm.covariances_[gg][dims_input,:][:,dims_input]) @ (X[nn,:] - gmm.means_[gg,dims_input] )

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

    def visualize_data_and_gaussians(self, n_grid=100, x_range=None, y_range=None):
        """ Visualize the results. """
        if x_range is None:
            x_range = [np.min(self.pos[:, 0]), np.max(self.pos[:,0])]
            
        if y_range is None:
            y_range = [np.min(self.pos[:, 1]), np.max(self.pos[:, 1])]
        
        xlim = [x_range[0]-1, x_range[1]+1]
        ylim = [y_range[0]-1, y_range[1]+1]
        
        nx, ny = n_grid, n_grid
        xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny))
        pos_x = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))

        output_gmm = self.regress_gmm(pos_x.T, self.dpgmm, self.dims_input,
                                      self.meanX, self.varX, attractor=self.pos_attractor)

        vel = get_angle_space_inverse_of_array(
            vecs_angle_space=output_gmm[:, :self.dim-1].T,
            positions=pos_x, func_vel_default=evaluate_linear_dynamical_system)

        x_traj = self.integrate_trajectory()

        print('Start creating plot.')
        plt.figure()
        plt.plot(self.pos[:,0], self.pos[:,1], '.b')
        # draw_gaussians(dpgmm, ax_time, [3,2])
        # plt.plot(pos[:,0], pos[:,1], '.k')

        for ii in range(x_traj.shape[2]):
            plt.plot(x_traj[0, 0, ii], x_traj[1, 0, ii], '.')
            plt.plot(x_traj[0, :, ii], x_traj[1, :, ii], '-.', linewidth=4)

        plt.streamplot(xGrid, yGrid, vel[0,:].reshape(nx, ny), vel[1,:].reshape(nx,ny),
                       color='blue')
        
        plt.axis('equal')
        plt.show()

        # plt.quiver()

    def plot_position_and_gaussians_2d(self, figure_name='position_and_gauss', save_figure=False):
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.plot(self.pos_attractor[0], self.pos_attractor[1], 'k*', markersize=12)
        plt.axis('equal')
        draw_gaussians(self.dpgmm, ax, [0,1])
        plt.plot(self.X[:,0], self.X[:,1], '.')
        plt.axis('equal')
        plt.xlabel(r'$\xi_1$')
        plt.ylabel(r'$\xi_2$')
        # plt.xlabel('x_1')
        # y_train_pred = estimator.predict(X_train)
        # train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        # plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 # transform=h.transAxes)

        # y_test_pred = estimator.predict(X_test)
        # test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        # plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 # transform=h.transAxes)

        # plt.title(name)
        if save_figure:
            plt.savefig(os.join.pay('figures', figure_name+".png", bbox_inches="tight"))

        
    def plot_time_direction_and_gaussians(self, figure_name="gmm_on_timeDirection", save_figure=False):
        # Plot of 
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.plot(self.X[:,3], self.X[:,2], '.')
        draw_gaussians(self.dpgmm, ax, [3,2])
        # plt.plot(t, dir[0], '.')
        plt.xlabel('Time [s/s]')
        plt.ylabel('Diection [rad/rad]')
        # NO equal axis due to different 'values'
        
        if save_figure:
            plt.savefig(os.join.pay('figures', figName+".png", bbox_inches='tight'))

    def plot_vector_field_weights(n_grid=100, x_range=None, y_range=None):
        """ Visualize the results. """
        if x_range is None:
            x_range = [np.min(self.pos[:, 0]), np.max(self.pos[:,0])]
            
        if y_range is None:
            y_range = [np.min(self.pos[:, 1]), np.max(self.pos[:, 1])]
        
        xlim = [x_range[0]-1, x_range[1]+1]
        ylim = [y_range[0]-1, y_range[1]+1]
        
        nx, ny = n_grid, n_grid
        xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny))
        position = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))

        output_gmm = self.regress_gmm(pos_x.T, self.dpgmm, self.dims_input,
                                      self.meanX, self.varX, attractor=self.pos_attractor)

        

        
        
