'''
SEDS adaptation

@author lukashuber
@date 2018-12-10
'''

import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion() # continue program when showing figures

from math import pi

import scipy.io # import *.mat files -- MATLAB files

import warnings

import random

# Machine learning datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import mixture

import sys

saveFigure = False

lib_string = "/home/lukas/Code/MachineLearning/SEDS_linear/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from lib_directionLearning import *

# Quadratic programming
from cvxopt.solvers import coneqp

showing_figures = True
# plt.close('all')
print('Start script .... \n\n\n')
# a_handwriting = scipy.io.loadmat('dataset/2D_Ashape.mat')

dataset = scipy.io.loadmat('dataset/2D_messy-snake.mat')
n_gaussian = 17

# dataset = scipy.io.loadmat('dataset/2D_incremental_1.mat')
# n_gaussian = 5

# dataset = scipy.io.loadmat('dataset/2D_Sshape.mat')
# n_gaussian = 5

# dataset = scipy.io.loadmat('dataset/2D_Ashape.mat')
# n_Gaussian = 6

# dataset = scipy.io.loadmat('dataset/2D_Ashape.mat')
# n_Gaussian = 6

dim_space = 2


colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red', 'green', 'purple', 'black', 'violet', 'tan']
# colors = ['navy']

def draw_gaussians(gmm, ax, plot_dims):
    # for n, color in enumerate(colors):
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


if False:
# if showing_figures: 
    plt.figure() # Plot initial dataset
    h = plt.subplot(1,1,1)
    i = 0
    for i in range(3):
        plt.plot(dataset['data'][0,i][0,:], dataset['data'][0,i][1,:], '.')
    figName = 'initialDataset'
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/SEDS_linear/fig/' + figName + '.eps', bbox_inches='tight')

ii = 0 # Only take the first fold.
pos = dataset['data'][0,ii][:2,:].T
vel = dataset['data'][0,ii][2:4,:].T
t = np.linspace(0,1,dataset['data'][0,ii].shape[1])
pos_attractor =  np.zeros((dim_space))

for it_set in range(1,dataset['data'].shape[1]):
    pos = np.vstack((pos, dataset['data'][0,it_set][:2,:].T))
    vel = np.vstack((vel, dataset['data'][0,it_set][2:4,:].T))

    pos_attractor = pos_attractor + dataset['data'][0,it_set][:2,-1].T/dataset['data'].shape[1]
    
    # TODO include velocity - rectify
    t = np.hstack((t, np.linspace(0,1,dataset['data'][0,it_set].shape[1])))

x_range = [np.min(pos[:,0]), np.max(pos[:,0])]
y_range = [np.min(pos[:,1]), np.max(pos[:,1])]
    
direction, rotMatrix = velocity_reduction(pos.T, vel.T, pos_attractor=pos_attractor)
 
a_label = np.zeros(pos.shape[0])


X = np.hstack((pos, direction.T, np.tile(t, (1,1)).T ))

n_samples = X.shape[0]
dim = X.shape[1]

weightDir = 4
# Normalize dataset
meanX = np.mean(X, axis=0)

meanX = np.zeros(4)
# X = X - np.tile(meanX , (X.shape[0],1))
varX = np.var(X, axis=0)

varX[:dim_space] = np.mean(varX[:dim_space]) # All distances should have same variance
varX[dim_space:2*dim_space-1] = np.mean(varX[dim_space:2*dim_space-1]) # All directions should have same variance
varX[dim_space:2*dim_space-1] = varX[dim_space:2*dim_space-1]*1/weightDir # stronger weight on directions!

# for ii in range(varX.shape[0]): # todo remove mean
    # varX[ii] = 1
    # meanX[ii] = 0

X = X/np.tile(varX , (X.shape[0],1))

# Choose starting values


pos_attractor = (pos_attractor-meanX[:dim_space]) / varX[:dim_space]
# X = X[:,:2]

# Split dataset test/train
 
tt_ratio = 0.75
all_index = np.arange(pos.shape[0])
train_index, test_index = train_test_split(all_index, test_size=(1-tt_ratio) )
X_train = X[train_index,:]
X_test = X[test_index,:]
y_train = a_label[train_index]
y_test = a_label[test_index]


cov_type = 'full'

if False: # 'Simple' gaussian regression
    plt.figure()
    plt.plot(X[:,0], X[:,1], '.')
    h = plt.subplot(1,1,1)
    estimator = GaussianMixture(n_components=n_gaussian,
                                covariance_type=cov_type, max_iter=300, random_state=0)
    estimator.means_init = X_train[np.random.randint(0,X_train.shape[0],n_gaussian),:]
    # estimator.means_init = np.array([X_train[y_train == ii,:].mean(axis=0)
                                # for i in range(n_classes)])
    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    draw_gaussians(estimator, h, [0,1])

    figName = 'regression_position'
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/SEDS_linear/fig/' + figName + '.eps', bbox_inches='tight')

# n_gaussian = 10
# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=n_gaussian, covariance_type='full')


# sample dataset
reference_dataset = 0
n_start = 0
for it_set in range(reference_dataset):
    n_start +=  dataset['data'][0,it_set].shape[1]
index_sample = [int(np.round(n_start+dataset['data'][0,reference_dataset].shape[1]/n_gaussian*ii)) for ii in range(n_gaussian)]

dpgmm.means_init = X[index_sample,:]
dpgmm.means_init = X_train[np.random.choice(np.arange(n_gaussian)),:]

dpgmm.fit(X_train[:,:])

if showing_figures:
    plt.figure()
    plotHandle_uniform = plt.subplot(1,1,1)
    plt.plot(pos_attractor[0], pos_attractor[1], 'k*', markersize=12)
    plt.axis('equal')
    draw_gaussians(dpgmm, plotHandle_uniform, [0,1])
    plt.plot(X[:,0], X[:,1], '.')
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
    figName = 'gmm_on_position'
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/SEDS_linear/fig/' + figName + '.eps', bbox_inches='tight')


if showing_figures:
    # Plot of 
    plt.figure()
    ax_time = plt.subplot(1,1,1)
    plt.plot(X[:,3], X[:,2], '.')
    draw_gaussians(dpgmm, ax_time, [3,2])
    # plt.plot(t, dir[0], '.')
    plt.xlabel('Time [s/s]')
    plt.ylabel('Diection [rad/rad]')
    # NO equal axis due to different 'values'
    figName = 'gmm_on_timeDirection'
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/SEDS_linear/fig/' + figName + '.eps', bbox_inches='tight')




prob_gauss = get_gaussianProbability(X, dpgmm)

# TODO ? Normalize guassian? with alpha val?
index_gauss = np.argmax(prob_gauss, axis=0)
index_sum = [np.sum(index_gauss==i) for i in range(prob_gauss.shape[0])]

# Get mean velocity
meanDir_gaussian = np.zeros((dim-1, n_gaussian)) # default zero direction (straight to attractor)
for gg in range(n_gaussian):
    if index_sum[gg]: # nonzero
        meanDir_gaussian[:, gg] = np.mean( X[index_gauss==gg,dim_space:2*dim_space-1], axis=0)
    else:
        warnings.warn('WARNING -- empty gaussian  g={}'.format(gg) )


# Create steramplot
# xlim = [-0.1,6]
# xlim = [-6.1,0.1]
# ylim = [-2,2]
# xlim = [-8.1,1.9]
# ylim = [-4,4]

xlim = [x_range[0]-1, x_range[1]+1]
ylim = [y_range[0]-1, y_range[1]+1]

n_grid = 100
nx, ny = n_grid, n_grid
xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny))
pos_x = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))

# Test values
# nx, ny = 1, 1
# pos_x = np.array([[3],[-3]])
# xGrid = pos_x[0,:].reshape(nx,ny)
# yGrid = pos_x[1,:].reshape(nx,ny)

dims_input = [0,1]


print('Calculate grid {}x{}'.format(nx,ny))
# output_gmm = regress_gmm(pos_x.T, dpgmm, dims_input, mu, variance)
output_gmm = regress_gmm(pos_x.T, dpgmm, dims_input, meanX, varX, attractor=pos_attractor)

# prob_gauss = get_gaussianProbability(pos.T, dpgmm)

# Get Gaussian function for each point
# for gg in range(n_gaussian):    #
    # delta_direction = guassian_function(pos, dpgmm.means_[gg, :dim], meanDir_gaussian[:,gg])

    # direction = direction +  prob_gauss[gg,:]*delta_direction  #

vel = velocity_reconstruction(pos_x, output_gmm[:,:dim_space-1])

# Adapt magnitude
# vel_mag =


def mag_linear_maximum(x, k=1, maxMag = 1.0):
    magnitude = LA.norm(x,axis=0)*k
    max_indeces = magnitude > maxMag
    if np.sum(max_indeces.shape[0]): #nonzero
        magnitude[max_indeces] = maxMag*np.ones(np.sum(max_indeces) )

    return magnitude
        
vel = vel * np.tile(mag_linear_maximum(pos_x ), (dim_space,1))



def rk4(dt, x, ds, x0=[0,0], k_f=1):
    x0 =np.array((x0))
    # k1
    xd = ds(x)*k_f
    k1 = dt*xd

    # k2
    xd = ds(x+0.5*k1)*k_f
    k2 = dt*xd

    # k3
    xd = ds(x+0.5*k2)*k_f
    k3 = dt*xd

    
    # k4
    xd = ds(x+k3)*k_f
    k4 = dt*xd

    # x final
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)

    return x

# Inegrate trajectories
dt = 0.02
intSteps = 200

nTraj = 3
print('Integrate {} trajectories for n={} steps with dt={}'.format(nTraj,intSteps,dt))

def ds_func(xx):
        #return obs_avoidance_func(x, xd, obs, attractor=xAttractor)
        output_gmm = regress_gmm(np.array([xx]), dpgmm, dims_input, meanX, varX, attractor=pos_attractor)
        
        vel = velocity_reconstruction(np.array([xx]).T, output_gmm[:,:dim_space-1])

        vel = vel * np.tile(LA.norm(xx, axis=0), (dim_space,1)) # Adapt magnitude
        
        return np.squeeze(vel)

x_traj = np.zeros((dim_space, intSteps, nTraj))
for ii in range(nTraj):
    x_traj[:,0,ii] = dataset['data'][0,ii][:2,0]
    # plt.plot(dataset['data'][0,i][0,:], dataset['data'][0,i][1,:], '.')    
    for nn in range(1,intSteps):
        x_traj[:,nn,ii] = rk4(dt, x_traj[:,nn-1,ii], ds_func)


print('Start creating plot.')
if True:
    # plt.figure(7,5)
    plt.figure()
    plt.plot(pos[:,0], pos[:,1], '.b')
    # draw_gaussians(dpgmm, ax_time, [3,2])
    # plt.plot(pos[:,0], pos[:,1], '.k')
    plt.streamplot(xGrid, yGrid, vel[0,:].reshape(nx, ny), vel[1,:].reshape(nx,ny))

    plt.xlim(xlim)
    plt.ylim(xlim)

    for ii in range(nTraj):
        plt.plot(x_traj[0,0,ii], x_traj[1,0,ii], '.')
        plt.plot(x_traj[0,:,ii], x_traj[1,:,ii], '-.', linewidth=4)
    
    # plt.quiver(xGrid, yGrid, vel[0,:].reshape(nx, ny), vel[1,:].reshape(nx,ny))
    plt.axis('equal')
    plt.show()

    figName = 'regression_time'
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/SEDS_linear/fig/' + figName + '.eps', bbox_inches='tight')


    

    # plt.quiver()

print('\n\n\n... script finished.')
