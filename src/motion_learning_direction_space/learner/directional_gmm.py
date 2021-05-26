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

# Custom libraries
# from motion_learning_direction_space.visualization.gmm_visualization import draw_gaussians
from motion_learning_direction_space.math_tools import rk4, rk4_pos_vel, mag_linear_maximum
from motion_learning_direction_space.direction_space import velocity_reduction, get_gaussianProbability, velocity_reconstruction, velocity_reduction, get_mixing_weights, get_mean_yx
from motion_learning_direction_space.learner.base import Learner
from motion_learning_direction_space.learner.visualizer import LearnerVisualizer

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system


class DirectionalGMM(LearnerVisualizer, Learner):
    """ Virtual class to learn from demonstration / implementation.
    The Gaussian mixture regression on the slides found in:
    http://calinon.ch/misc/EE613/EE613-slides-9.pdf
    """
    def __init__(self, directory_name='', file_name='', dim_space=2):
        self.directory_name = directory_name
        self.file_name = file_name

        self.dim_space = self.dim = dim_space
        self.dim_gmm = None

        # TODO: remove dataset from 'attributes'
        self.dataset = None
        self.pos = None
        self.vel = None

        # Noramlized etc. regression data
        self.X = None

        super().__init__()

    @property
    def n_gaussians(self):
        return self.dpgmm.covariances_.shape[0]

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

    def velocity_actual_to_scaled(self, velocity):
        """ Scale velocity to align with the paremters.
        Currently no velocity factor is implemented.
        This means that the distance space influence is aproximately equivalent
        to being 2 meters apart. """
        if self.mean_velocity is None:
            return velocity

        # Scale by the mean & cap at 1
        vel_fac = np.linalg.norm(self.vel, axis=0)
        vel_fac = np.maximum(vel_fac, self.mean_velocity*np.zeros(vel_fac.shape))
        vel_fac = 1.0 / vel_fac

        velocity = velocity * np.tile(vel_fac, (velocity.shape[1], 1)).T

        # Scale by the mean & cap at 1
        return velocity

    def velocity_scaled_to_actual(self, velocity):
        """ Reverse the scaling of the velocity and returns."""
        raise NotImplementedError("Scaling back is not yet desired"
                                  + "additionally the minimum is a bijective funtion.")
    
    def normalize_velocity(self, velocity):
        """ Normalize the velocity by the mean & cap it at 1.0 """
        mean_vel = np.mean(np.linalg.norm(velocity, axis=0))
        velocity = velocity / mean_vel
        velocity = np.minimum(velocity, np.ones(velocity.shape))

        # Story mean_vel 
        self.mean_velocity = mean_vel
        return velocity
        
    def load_data_from_mat(self, file_name=None, feat_in=None):
        """ Load data from file mat-file & evaluate specific parameters """
        if file_name is not None:
            self.file_name = file_name

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name))

        if feat_in is None:
            self.feat_in = [0,1]
        
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

        self.vel = self.normalize_velocity(self.vel)
    
        self.X = np.hstack((self.pos, self.vel, direction.T))
        self.num_samples = self.X.shape[0]
        self.dim_gmm = self.X.shape[1]
        
        weightDir = 4
        
        # Normalize dataset
        normalize_dataset = False
        if normalize_dataset:
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
            
        else:
            self.meanX = None
            self.varX = None

            self.pos_attractor = pos_attractor
            

    def regress(self, *args, **kwargs):
        # TODO: remove
        return self.fit(*args, **kwargs)

    def fit(self, n_gaussian=5, tt_ratio=0.75):
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
            n_start +=  self.dataset['data'][0, it_set].shape[1]
            
        index_sample = [int(np.round(n_start + self.dataset['data'][0, reference_dataset].shape[1]
                                     /n_gaussian*ii)) for ii in range(n_gaussian)]

        self.dpgmm.means_init = self.X[index_sample, :]
        self.dpgmm.means_init = X_train[np.random.choice(np.arange(n_gaussian)), :]

        self.dpgmm.fit(X_train[:, :])

    
    def regress_gmm(self, *args, **kwargs):
        # TODO: Depreciated; remove..
        return self._predict(*args, **kwargs)

    def _predict(self, X, input_output_normalization=True, feat_in=None, feat_out=None,
                    convergence_attractor=True, p_beta=2, beta_min=0.5, beta_r=0.3):
        """ Evaluate the regress field at all the points X""" 
        # output_gmm = self.regress_gmm(pos_x.T, self.dpgmm, self.self.feat_in,
                                      # self.meanX, self.varX, attractor=self.pos_attractor)
        dim = self.dim_gmm
        n_samples = X.shape[0]
        dim_in = X.shape[1]
        
        if feat_in is None:
            feat_in = np.arange(dim_in)

        if feat_out is None:
            # Default only the 'direction' at the end; additional -1 for indexing at end
            feat_out = self.dim_gmm - 1 - np.arange(self.dim_space-1)
        dim_out = np.array(feat_out).shape[0]

        if input_output_normalization:
            X = self.transform_initial_to_normalized(X, dims_ind=feat_in)

        # Gausian Mixture Model Properties
        beta = self.get_mixing_weights(X, feat_in=feat_in, feat_out=feat_out)
        mu_yx = self.get_mean_yx(X, feat_in=feat_in, feat_out=feat_out)
        
        if convergence_attractor:
            if self.pos_attractor is None:
                raise ValueError("Convergence to attractor without attractor...")
            dist_attr = np.linalg.norm(
                X - np.tile(self.pos_attractor, (n_samples, 1)) , axis=1)

            beta = np.vstack((beta, np.zeros(n_samples)))

            # Zero values
            ind_zero = dist_attr==0
            beta[:, ind_zero] = 0
            beta[-1, ind_zero] = 1

            # Nonzeros values
            ind_nonzero = dist_attr!=0 
            beta[-1, ind_nonzero] =  (dist_attr[ind_nonzero]/beta_r)**(-p_beta) + beta_min 
            beta[:, ind_nonzero] = (beta[:, ind_nonzero] / np.tile(
                np.linalg.norm(beta[:, ind_nonzero], axis=0), (self.n_gaussians+1, 1)))

            mu_yx = np.dstack((mu_yx, np.zeros((dim_out, n_samples,1))))

        regression_value = np.sum(np.tile(beta.T, (dim_out, 1, 1) ) * mu_yx, axis=2).T

        if input_output_normalization:
            regression_value = self.transform_normalized_to_initial(
                regression_value, dims_ind=feat_out)
            
        return regression_value

    def get_mixing_weights(self, X, feat_in, feat_out, input_needs_normalization=False,
                           normalize_probability=False, weight_factor=15.0):
        """ Get input positions X of the form [dimension, number of samples]."""
        # TODO: try to learn the 'weight_factor' [optimization problem?]
        if input_needs_normalization:
            X = self.transform_initial_to_normalized(X, feat_in)
            
        n_samples = X.shape[0]

        prob_gaussian = self.get_gaussian_probability(X, feat_in=feat_in)
        sum_probGaussian = np.sum(prob_gaussian, axis=0)

        alpha_times_prob = np.tile(self.dpgmm.weights_, (n_samples, 1)).T  * prob_gaussian

        if normalize_probability:
            beta = alpha_times_prob / np.tile(np.sum(alpha_times_prob, axis=0),
                                              (self.n_gaussians, 1))
        else:
            beta = alpha_times_prob
            # *weight_fac
            beta = beta**(1./1)
            max_weight = np.max(self.dpgmm.weights_)
            beta = beta/max_weight * weight_factor

            sum_beta = np.sum(beta, axis=0)
            ind_large = sum_beta > 1
            beta[:, ind_large] = beta[:, ind_large] / np.tile(sum_beta[ind_large],
                                                              (self.n_gaussians, 1))

        return beta

    def get_gaussian_probability(self, X, feat_in):
        """ Returns the array of 'mean'-values based on input positions.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.
    
        Returns
        -------
        prob_gauss (beta): array of shape (n_samples)
            The weights (similar to prior) which is gaussian is assigned.
        """
        n_samples = X.shape[0]

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((self.n_gaussians, n_samples))

        for gg in range(self.n_gaussians):
            # Create function of this
            cov_matrix = self.dpgmm.covariances_[gg, :, :][self.feat_in, :][:, self.feat_in]
            fac = 1/((2*pi)**(self.dim*.5)*(np.linalg.det(cov_matrix))**(0.5))

            dX = X - np.tile(self.dpgmm.means_[gg, self.feat_in], (n_samples,1))

            val_pow = np.sum(np.tile(np.linalg.pinv(cov_matrix), (n_samples, 1, 1))
                             *np.swapaxes(np.tile(dX,  (self.dim, 1, 1)), 0, 1), axis=2)
            
            val_pow = np.exp(-np.sum(dX*val_pow, axis=1))
            prob_gauss[gg, :] = fac*val_pow
        return prob_gauss

    def get_mean_yx(self, X, feat_in, feat_out, stretch_input_values=False):
        """ Returns the array of 'mean'-values based on input positions.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.
    
        Returns
        -------
        mu_yx: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        dim = self.dim_gmm
        dim_in = np.array(feat_in).shape[0]
        dim_out = np.array(feat_out).shape[0]

        n_samples = X.shape[0]

        mu_yx = np.zeros((dim_out, n_samples, self.n_gaussians))
        mu_yx_hat = np.zeros((dim_out, n_samples, self.n_gaussians))

        for gg in range(self.n_gaussians):
            mu_yx[:, :, gg] = np.tile(self.dpgmm.means_[gg, feat_out], (n_samples, 1)).T
            matrix_mult = self.dpgmm.covariances_[gg][feat_out, :][:, self.feat_in].dot(
                np.linalg.pinv(self.dpgmm.covariances_[gg][feat_in, :][:, self.feat_in]))
            mu_yx[:, :, gg] += matrix_mult.dot((
                X - np.tile(self.dpgmm.means_[gg, feat_in], (n_samples, 1))).T)

            # START REMOVE
            covariance_inverse = np.linalg.pinv(
                self.dpgmm.covariances_[gg][feat_in, :][:, feat_in])

            covariance_output_input = (
                self.dpgmm.covariances_[gg][feat_out, :][:, feat_in])
            for nn in range(n_samples): # TODO #speed - batch process!!
                mu_yx_hat[:, nn, gg] = (
                    self.dpgmm.means_[gg, feat_out]
                    + covariance_output_input
                    @ covariance_inverse
                    @ (X[nn, :] - self.dpgmm.means_[gg, feat_in])
                )

            if np.sum(mu_yx-mu_yx_hat) > 1e-6:
                breakpoint()
            else:
                #TODO: remove when warning never shows up anymore
                warnings.warn("Remove looped multiplication, since is the same...")
        return mu_yx

    def integrate_trajectory(self, num_steps=200, delta_t=0.02, nTraj=3, starting_points=None,
                             convergence_err=0.01):
        """ Return integrated trajectory with runge-kutta-4 based on the learned system.
        Default starting points are chosen at the starting points of the learned data"""
        
        if starting_points is None:
            x_traj = np.zeros((self.dim, num_steps, nTraj))
            for ii in range(nTraj):
                x_traj[:, 0, ii] = self.dataset['data'][0, ii][:2, 0]
        else:
            nTraj = starting_points.shape[1]
            x_traj = np.zeros((self.dim, num_steps, nTraj))

        # Do the first step without velocity
        print("Doint the integration.")
        for ii in range(nTraj):
            x_traj[:, 1, ii]= rk4(
                    delta_t, x_traj[:, nn-1, ii], self.predict)
            
            xd = x_traj[:, 1, ii] - x_traj[:, 0, ii] 
            for nn in range(2, num_steps):
                x_traj[:, nn, ii], xd = rk4_pos_vel(
                    dt=delta_t, pos0=x_traj[:, nn-1, ii], vel0=xd,
                    ds=self.predict)
                
                if np.linalg.norm(x_traj[:, nn, ii]) < convergence_err:
                    print(f"Converged after {nn} iterations.")
                    break
                
        print("This took me a while...")
        return x_traj

