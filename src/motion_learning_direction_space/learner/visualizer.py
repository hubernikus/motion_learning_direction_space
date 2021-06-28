"""
Visualizer Class to add-on to the Learner-subclassses
"""

# Author:  Lukas Huber
# Mail: hubernikus@gmail.com
# Created: 2021-05-24
# License: BSD (c) 2021

from math import pi
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

class LearnerVisualizer():
    """ All the visualization function for use across different learning models. """
    def plot_position_data(self, x_lim=None, y_lim=None, ax=None):
        x_lim, y_lim = self.get_xy_lim_plot()

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
        
        ax.plot(self.pos[:,0], self.pos[:,1], '.', color='blue', markersize=1)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def plot_vectorfield_and_integration(self, n_grid=100, x_range=None, y_range=None):
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

        vel = self.predict_array(pos_x)
        x_traj = self.integrate_trajectory()

        print('Start creating plot.')
        plt.figure()
        plt.plot(self.pos[:,0], self.pos[:,1], '.', color='black')
        # self.draw_gaussians(self.dpgmm, ax_time, [3,2])
        # plt.plot(pos[:,0], pos[:,1], '.k')

        for ii in range(x_traj.shape[2]):
            plt.plot(x_traj[0, 0, ii], x_traj[1, 0, ii], '.')
            plt.plot(x_traj[0, :, ii], x_traj[1, :, ii], '-.', linewidth=4)

        plt.streamplot(xGrid, yGrid, vel[0,:].reshape(nx, ny), vel[1,:].reshape(nx,ny),
                       color='blue')
        
        plt.axis('equal')
        plt.show()

        # plt.quiver()

    def plot_position_and_gaussians_2d(self, figure_name='position_and_gauss', save_figure=False, colors=None, ):
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.plot(self.pos_attractor[0], self.pos_attractor[1], 'k*', markersize=12)
        plt.axis('equal')
        self.draw_gaussians(self.dpgmm, ax, [0,1], colors=colors)
        
        # plt.plot(self.X[:,0], self.X[:,1], '.')
        plt.plot(self.pos[:,0], self.pos[:,1], '.')

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

        # Set range to easier compare
        x_range = [np.min(self.pos[:, 0]), np.max(self.pos[:,0])]
        y_range = [np.min(self.pos[:, 1]), np.max(self.pos[:, 1])]
        xlim = [x_range[0]-1, x_range[1]+1]
        ylim = [y_range[0]-1, y_range[1]+1]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # plt.title(name)
        if save_figure:
            plt.savefig(os.join.pay('figures', figure_name+".png", bbox_inches="tight"))

    def plot_gaussians_all_directions(self):
        """ Plot Gaussians in All Directions. """ 
        colorlist = self.complementary_color_picker(self.n_gaussians)

        for ii in range(self.X.shape[1]):
            for jj in range(ii, self.X.shape[1]):
                if ii == jj:
                    # Do special plot
                    continue
                it = 1 + (ii) + jj*self.X.shape[1]
                ax1 = plt.subplot(self.X.shape[1], self.X.shape[1], it)
                self.draw_gaussians(self.dpgmm, ax1, [ii, jj], colors=colorlist)
                plt.plot(self.X[:, ii], self.X[:, jj], 'k.', markersize=1)
                plt.xlabel(f"{ii}"), plt.ylabel(f"{jj}")
                
                it = 1 + (jj) + ii*self.X.shape[1]
                ax2 = plt.subplot(self.X.shape[1], self.X.shape[1], it)
                self.draw_gaussians(self.dpgmm, ax2, [jj, ii], colors=colorlist)
                plt.plot(self.X[:, jj], self.X[:, ii], 'k.', markersize=1)
                plt.xlabel(f"{jj}"), plt.ylabel(f"{ii}")
                
        
    def plot_time_direction_and_gaussians(self, figure_name="gmm_on_timeDirection",
                                          save_figure=False, colors=None):
        # Plot of 
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.plot(self.X[:,3], self.X[:,2], '.')
        self.draw_gaussians(self.dpgmm, ax, [3,2], colors=None)
        # plt.plot(t, dir[0], '.')
        plt.xlabel('Time [s/s]')
        plt.ylabel('Diection [rad/rad]')
        # NO equal axis due to different 'values'
        
        if save_figure:
            plt.savefig(os.join.pay('figures', figName+".png", bbox_inches='tight'))

    def get_xy_lim_plot(self, margin_x=1, margin_y=1):
        """ Returns xlim & ylim of the data (plus/minus margin) in order to plot it nicely """
        x_range = [np.min(self.pos[:, 0]), np.max(self.pos[:,0])]
        y_range = [np.min(self.pos[:, 1]), np.max(self.pos[:, 1])]
        
        xlim = [x_range[0]-margin_x, x_range[1]+margin_x]
        ylim = [y_range[0]-margin_y, y_range[1]+margin_y]

        return xlim, ylim
        
    def plot_vector_field_weights(self, n_grid=100, xlim=None, ylim=None, colorlist=None, pos_vel_input=False):
        """ Visualize the results. """
        if xlim is None:
            xlim, ylim = self.get_xy_lim_plot()
        
        nx, ny = n_grid, n_grid
        xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny))
        positions = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))
        velocities = self.predict_array(positions)
        
        if pos_vel_input:
            pos_vel = np.vstack((positions, velocities))
            weights = self.get_mixing_weights(pos_vel.T, input_needs_normalization=True,
                                              feat_in=np.array([0, 1, 2, 3]), feat_out=np.array([-1]))
        else:
            weights = self.get_mixing_weights(positions.T, input_needs_normalization=True,
                                              feat_in=np.array([0, 1]), feat_out=np.array([-1]))

        if colorlist is None:
            colorlist = self.complementary_color_picker(n_colors=self.n_gaussians, offset=0)
        
        # Do negative color
        # rgb_image = np.zeros((n_grid, n_grid, colorlist.shape[0]))
        rgb_image = np.ones((n_grid, n_grid, colorlist.shape[0]))
        for ii in range(self.n_gaussians):
            weight_gauss = weights[ii, :].reshape(n_grid, n_grid)
            for cc in range(colorlist.shape[0]):
                rgb_image[:, :, cc] = (rgb_image[:, :, cc] 
                                       + weight_gauss*colorlist[cc, ii] - weight_gauss)
        
        # colorlist_expand = np.swapaxes(np.tile(colorlist, (weights.shape[1], 1, 1)), 1, 2)
        # weights_expand = np.swapaxes(np.tile(weights, (colorlist.shape[0], 1, 1)), 0, 2)
        # rgb_list = np.sum(colorlist_expand*weights_expand, axis=1)

        # rgb_image = np.zeros((n_grid, n_grid, rgb_list.shape[1]))
        # for ii in range(rgb_list.shape[1]):
            # rgb_image = rgb_list[:, ii].reshape(n_grid, n_grid)

        plt.figure()
        plt.imshow(rgb_image, zorder=-3, 
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

        plt.plot(self.pos[:,0], self.pos[:,1], '.', color='black')

        plt.streamplot(xGrid, yGrid,
                       velocities[0,:].reshape(nx, ny), velocities[1,:].reshape(nx,ny),
                       # color='white')
                       color='#808080')

        # self.plot_weight_subplots(n_grid=n_grid, colorlist=colorlist, weights=weights)


    def plot_weight_subplots(self, n_grid, colorlist=None, weights=None):
        xlim, ylim = self.get_xy_lim_plot()

        if colorlist is None:
            # Do several suplots
            colorlist = self.complementary_color_picker(
                n_colors=self.n_gaussians, offset=0)
        if weights is None:
            nx, ny = n_grid, n_grid
            xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx), np.linspace(ylim[0], ylim[1], ny))
            positions = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))

            weights = self.get_mixing_weights(positions.T, input_needs_normalization=True,
                                              feat_in=np.array([0, 1]), feat_out=np.array([-1]))
            
        fig, axs = plt.subplots(1, self.n_gaussians, figsize=(12, 3))
        for ii in range(self.n_gaussians):
            weight_gauss = weights[ii, :].reshape(n_grid, n_grid)

            rgb_image = np.zeros((n_grid, n_grid, colorlist.shape[0]))
            for cc in range(colorlist.shape[0]):
                rgb_image[:, :, cc] = weight_gauss*colorlist[cc, ii]
                
            axs[ii].imshow(rgb_image,
                           extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

    def draw_gaussians(self, gmm, ax, plot_dims, colors=None, edge_only=False):
        if colors is None:
            colors = ['navy', 'turquoise', 'darkorange', 'blue',
                      'red', 'green', 'purple', 'black', 'violet', 'tan']

        if isinstance(colors, np.ndarray):
            color_list = []
            for ii in range(colors.shape[1]):
                color_list.append(colors[:, ii])
            colors = color_list

        # for n, color in enumerate(colors):
        n_gaussian = gmm.n_components
        for n in range(n_gaussian):
            color = colors[np.mod(n, len(colors))]
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
            angle = 180 * angle / np.pi  # Convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            # ell = mpl.patches.Ellipse(gmm.means_[n, plot_dims], v[0], v[1], 180 + angle, color=color)

            # Stretch along the main axes
            if self.varX is not None:
                means_value = gmm.means_[n, plot_dims] * self.varX[plot_dims]
                v = v * self.varX[plot_dims]
            else:
                means_value = gmm.means_[n, plot_dims]

            if edge_only:
                ell = mpl.patches.Ellipse(means_value, v[0], v[1], 180 + angle,
                                          edgecolor='k', alpha=1, fill=False)
            else:
                ell = mpl.patches.Ellipse(means_value, v[0], v[1], 180 + angle, color=color)

            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.plot(means_value[0], means_value[1], 'k.', markersize=12, linewidth=30)
            # ax.plot(gmm.means_[n, 0], gmm.means_[n, 1], 'k+', s=12)

    def plot_relative_direction(self, n_grid=100, xlim=None, ylim=None):
        self.plot_vectorfield_and_integration(n_grid=n_grid)
        
        if xlim is None:
            xlim, ylim = self.get_xy_lim_plot()
        
        nx, ny = n_grid, n_grid
        xGrid, yGrid = np.meshgrid(np.linspace(xlim[0], xlim[1], nx),
                                   np.linspace(ylim[0], ylim[1], ny))
        positions = np.vstack((xGrid.reshape(1,-1), yGrid.reshape(1,-1)))
        # velocities = self.predict_array(positions)
        directions_angle_space = self._predict(positions.T)
        # direction = self.predict_array(positions)

        plt.contourf(xGrid, yGrid,
                     np.reshape(directions_angle_space, (n_grid, n_grid)), 20, cmap='RdGy')

        min_val = np.min(directions_angle_space)
        max_val = np.min(directions_angle_space)

        limit_val = max(abs(min_val), abs(max_val))
        
        plt.clim(-limit_val, limit_val)
        plt.colorbar()

    @staticmethod
    def complementary_color_picker(n_colors=5, offset=0):
        delta_angle = 2*pi/n_colors
        angle_list = np.arange(n_colors)*delta_angle

        # Store the rgb colors for each point
        colors = np.zeros((3, n_colors))

        rgb_shift = 2*pi * 1./3

        angle = 0
        for ii, angle in zip(range(n_colors), angle_list):
            if angle < 1*rgb_shift:
                colors[0, ii] = (rgb_shift - angle) / rgb_shift
                colors[1, ii] = 1 - colors[0, ii]
                
            elif angle < 2*rgb_shift:
                colors[1, ii] = (2*rgb_shift-angle) / rgb_shift
                colors[2, ii] = 1 - colors[1, ii]
                
            else:
                colors[2, ii] = (3*rgb_shift-angle) / rgb_shift
                colors[0, ii] = 1 - colors[2, ii]

        # Make sure that all colors are wihtin [0, 1]
        colors = np.maximum(0, colors)
        colors = np.minimum(1, colors)

        # Offset deactivated / not working
        if True:
            return colors
        if offset > 0:   # Nonzero
            colors = np.minimum(colors + np.ones(colors.shape)*offset, np.ones(angle.shape))
        elif offset < 0:
            colors = np.maximum(colors + np.ones(colors.shape)*offset, np.zeros(angle.shape))
