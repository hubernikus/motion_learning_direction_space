"""
The learning.graph includes elements to automatically build graphs of GMM's to
simplify the model creation
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np
import matplotlib.pyplot as plt

from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_angle_space_inverse_of_array
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer

from motion_learning_direction_space.learner.directional_gmm import DirectionalGMM


class GraphGMM(MultiBoundaryContainer):
    """ Creats grpah from input gmm
    
    The main idea is (somehow to an overcompetivtive faimliy):
    - Direct-successor are friends (Grand-grand-...-parents to grand-grand-...-child)
    - Sibilings (plus cousins and all not successors) are rivals

    This is additionally an obstacle container (since graph) with multiple hulls.
    No inheritance from BaseContainer due to desired difference in behavior as
    this class has a graph-like structure.
    """
    def __init__(self, file_name, n_gaussian,
                 LearnerType=DirectionalGMM, *args, **kwargs):
        # Ellipse factor relative to 
        self._ellipse_axes_factor = 1

        # End point is the the direction of intersection
        self._end_points = None
        self._graph_root = None

        self._obstacle_list = None

        self._Learner = DirectionalGMM()
        self._Learner.load_data_from_mat(file_name=file_name)
        self._Learner.regress(n_gaussian=n_gaussian)
        # breakpoint()

    @property
    def gmm(self):
        return self._Learner.dpgmm

    @property
    def n_gaussians(self):
        return self._Learner.dpgmm.covariances_.shape[0]

    @property
    def dim_space(self):
        return self._Learner.dim_space

    @property
    def null_ds(self):
        # DS going to the attractor
        return self._Learner.null_ds

    @property
    def pos_attractor(self):
        # DS going to the attractor
        return self._Learner.pos_attractor

    @property
    def _attractor_position(self):
        # DS going to the attractor
        return self._Learner.pos_attractor

    @_attractor_position.setter
    def _attractor_position(self, value):
        # DS going to the attractor
        self._Learner.pos_attractor = value

    def get_convergence_direction(self, position, it_obs):
        """ Get the (null) direction for a specific gaussian-hull in the multi-body-boundary
        container which serves for the rotational-modulation.
        
        The direction is based on a locally linear dynamical-system. """
        # Check if attractor is in current object
        if self[it_obs].get_gamma(self.pos_attractor, in_global_frame=True) >= 1:
            local_attractor = self.pos_attractor
        else:
            # Otherwise use the 'connection point' [which was chosen as global connection]
            local_attractor = self[it_obs].get_intersection_with_surface(
                direction=(self._end_points[:, it_obs]-self[it_obs].center_position),
                in_global_frame=True)

        return  evaluate_linear_dynamical_system(
            position=position, center_position=local_attractor)

    def get_xy_lim_plot(self):
        """ Return (x_lim, y_lim) tuple based on recorded dataset. """
        return self._Learner.get_xy_lim_plot()

    def get_mixing_weights(self, *args, **kwargs):
        return self._Learner.get_mixing_weights(*args, **kwargs)

    def ellipse_axes_length(self, it, axes_factor=3):
        """ Get axes length of ellipses extracted from the GMM-covariances. """
        # Get intersection with circle and then ellipse hull
        if self.gmm.covariance_type == 'full':
            covariances = self.gmm.covariances_[it, :, :][:self.dim_space, :][:, :self.dim_space]
        else:
            raise TypeError("Not implemented for unfull covariances")
        covariances = covariances * axes_factor**self.dim_space
        return covariances

    def get_end_point(self, it):
        """ Return the end point of a specific gaussian parameter.

        Note: this is (simplified) since we assume the orienation being the
        same as it is at the center. """
        
        mean = self.gmm.means_[it, :]

        mean_pos = mean[:self.dim_space]
        mean_dir = mean[-(self.dim_space-1):]

        null_direction = self.null_ds(mean_pos)

        center_velocity_dir = get_angle_space_inverse(
            dir_angle_space=mean_dir, null_direction=null_direction)

        if self.dim_space > 2:
            raise NotImplementedError()
        # 2D only (!) -- temporary; exand this!
        covariances = self.gmm.covariances_[it][:self.dim_space,:][:, :self.dim_space]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        # angle = 180 * angle / np.pi      # Convert to degrees
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        v = np.sqrt(2.) * np.sqrt(v)

        angle = -angle
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])

        axes_lengths = self.ellipse_axes_length(it=it, axes_factor=self._ellipse_axes_factor)
        end_point = mean_pos + axes_lengths.dot(center_velocity_dir)

        center_velocity_dir = rot_mat.T @ center_velocity_dir
        fac = np.sum(center_velocity_dir**2 / v**2)
        center_velocity_dir = center_velocity_dir/np.sqrt(fac)
        # center_velocity_dir = center_velocity_dir * v
        center_velocity_dir = rot_mat @ center_velocity_dir

        end_point = mean_pos + center_velocity_dir
        return end_point

    def get_rival_weight(self, ):
        """ """
        end_point = np.arange(3)
        pass

    def _get_assigned_elements(self):
        """ Get the list of all assigned elements.
        This is useful only during of the graph.""" 
        if self._graph_root is None:
            return []
        graph = [self._graph_root]
        for sub_list in self._children_list:
            graph = graph + sub_list 
        return graph

    @property
    def number_of_levels(self):
        raise NotImplementedError()
        # return -1

    @property
    def number_of_branches(self):
        """ Returns the number of brances by counting the number of 'dead-ends'."""
        return np.sum([not(len(list)) for list in self._children_list])

    def create_graph_from_gaussians(self):
        """ Create graph from learned GMM."""
        self._end_points = np.zeros((self.dim_space, self.n_gaussians))
        self._parent_array = (-1)*np.ones(self.n_gaussians, dtype=int)
        self._children_list = [[] for ii in range(self.n_gaussians)]

        # First 'level' is based on closest to origin
        for ii in range(self.n_gaussians):
            self._end_points[:, ii] = self.get_end_point(it=ii)
            
        # Get root / main-parent (this could be replaced by optimization / root finding)
        dist_atrractors = np.linalg.norm(
            self._end_points-np.tile(self.pos_attractor, (self.n_gaussians, 1)).T, axis=0)
            
        self._graph_root = np.argmin(dist_atrractors)

        weights = self.get_mixing_weights(
            X=self._end_points.T, weight_factor=self.dim_space,
            feat_in=np.arange(self.dim_space), feat_out=-np.arange(self.dim_space, 1)).T

        weights_without_self = np.copy(weights)
        for gg in range(self.n_gaussians):
            weights_without_self[gg, gg] = -1
            
        parents_preference = np.argsort(weights_without_self, axis=1)
        parents_preference = np.flip(parents_preference, axis=1)

        it_count = 0
        while True:
            it_count += 1
            if it_count > 100:
                # TODO: remove it_count (only for debugging purposes)
                breakpoint()
            ind_assigned = self._get_assigned_elements()
            if len(ind_assigned) == self.n_gaussians:
                break
            ind_unassigned = [ii for ii in range(self.n_gaussians) if ii not in ind_assigned]

            print('ind_assigned', ind_assigned)

            for it_pref in range(self.n_gaussians-1):
                list_parent_child = []
                for ind_child in ind_unassigned:
                    # TODO: to speed up only look at last index (new parents),
                    # when the last run fully resolved
                    if (parents_preference[ind_child, it_pref] in ind_assigned):
                        list_parent_child.append(
                            (parents_preference[ind_child, it_pref], ind_child))

                print(f'List at {ii} is {list_parent_child}')
                if len(list_parent_child):
                    # If it_pref > 0 it means that the prefered parent choice would be another one
                    # in that case the prefered graph might be a different one (!)
                    # Hence it will be double checked that this is actually the optimal one!
                    
                    if it_pref and len(list_parent_child) > 1:
                        # If the chosen one is the secondary choice, only adapt one
                        # and then reitarate, i.e. make an optimal sequence / graph
                        ind_critical = [child for par, child in list_parent_child]
                        ind_child = np.argmax(weights_without_self[ind_critical, it_pref])
                        ind_child = ind_critical[ind_child]
                        
                        self.extend_graph(parent=parents_preference[ind_child, it_pref],
                                          child=ind_child)
                        # TODO: Extesnively test this... Make sure it's working correctly.
                    else:
                        # Assign all elements if it's the first choice
                        for ind_parent, ind_child in list_parent_child:
                            self.extend_graph(child=ind_child, parent=ind_parent)
                    break

    def create_learned_boundary(self, oversize_factor=2.0):
        """ Adapt (inflate) each gaussian-ellipse to the graph such that there is a overlap.
        The simple gaussians are transformed to 'Obstacles'"""

        self._obstacle_list = []

        if self.dim_space != 2:
            # 2D only (!) -- temporary; exand this!
            raise NotImplementedError()
        
        for gg in range(self.n_gaussians):
            covariances = self.gmm.covariances_[gg][:self.dim_space,:][:, :self.dim_space]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            v = np.sqrt(2.) * np.sqrt(v)

            # Don't consider them as boundaries (yet)
            self._obstacle_list.append(
                Ellipse(
                center_position=self.gmm.means_[gg, :self.dim_space],
                orientation=angle,
                axes_length=v*oversize_factor,
                is_boundary=True,
                ))
            
        prop_dist_end = np.zeros(self.n_gaussians)
        
        for ii in range(self.n_gaussians):
            it_parent = self.get_parent(ii)
            prop_dist_end[ii] = self._obstacle_list[it_parent].get_gamma(
                self._end_points[:, ii], in_global_frame=True)
            pass

    def plot_obstacle_wall_environment(self):
        """Plot the environment such that we have an 'inverse' obstacle avoidance. """
        x_lim, y_lim = self._Learner.get_xy_lim_plot()

        plt.figure()
        ax = plt.subplot(1, 1, 1)

        boundary_patch = np.array([[x_lim[0], x_lim[1], x_lim[1], x_lim[0]],
                                   [y_lim[0], y_lim[0], y_lim[1], y_lim[1]]])
        boundary_patch = boundary_patch.T
        boundary_polygon = plt.Polygon(boundary_patch, alpha=1.0, zorder=-10)
        boundary_polygon.set_color(np.array([176, 124, 124])/255.)
        ax.add_patch(boundary_polygon)

        level_number = self.get_level_numbers()
        
        obs_polygon = []
        for it_obs, obs in zip(range(len(self._obstacle_list)), self._obstacle_list):
            obs.draw_obstacle()
            # Create boundary points
            obs_boundary = obs.boundary_points_global_closed
            obs_polygon.append(plt.Polygon(obs_boundary.T, alpha=1.0, zorder=-9))
                                           
            obs_polygon[-1].set_color(np.array([1.0, 1.0, 1.0]))
            ax.add_patch(obs_polygon[-1])

            ax.plot(obs_boundary[0, :], obs_boundary[1, :], '--',
                    color='k', zorder=-8, alpha=0.5)

            ax.plot(obs.center_position[0], obs.center_position[1], '+', color='k',
                    linewidth=18, markeredgewidth=4, markersize=13, zorder=-8)

            ax.plot(self._end_points[0, it_obs], self._end_points[1, it_obs], 'ro')

            local_attractor = self[it_obs].get_intersection_with_surface(
                direction=(self._end_points[:, it_obs]-self[it_obs].center_position),
                in_global_frame=True)
            ax.plot(local_attractor[0], local_attractor[1], 'r*')

            ax.plot([self._end_points[0, it_obs], local_attractor[0]],
                    [self._end_points[1, it_obs], local_attractor[1]], 'r')

            ax.annotate('{}'.format(level_number[it_obs]),
                        xy=obs.center_position+0.08,
                        textcoords='data', size=16, weight="bold")

        # Attractor and points
        ax.plot(self.pos_attractor[0], self.pos_attractor[1], 'k*', markersize=12)
        ax.plot(self._Learner.pos[:,0], self._Learner.pos[:,1], '.', color='blue', markersize=1)
        ax.axis('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def plot_graph_and_gaussians(self, colors=None, ax=None):
        """ Plot the graph and the gaussians as 'grid'."""
        x_lim, y_lim = self._Learner.get_xy_lim_plot()
        
        if colors is None:
            gauss_colors = self._Learner.complementary_color_picker(n_colors=self.n_gaussians)

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
        self.draw_gaussians(self.gmm, ax, [0,1], edge_only=True)
        # self.plot_position_and_gaussians_2d(colors=gauss_colors, edge_only=True)

        # Draw graph
        plt.plot(self._end_points[0, :], self._end_points[1, :], '+', color='red')
        center_positions = self.gmm.means_.T
        for ii in range(self.n_gaussians):
            plt.plot([self._end_points[0, ii], center_positions[0, ii]],
                     [self._end_points[1, ii], center_positions[1, ii]], '--', color='red')

            ind_parent = self.get_parent(ii)
            if ind_parent == (-1):
                # Connect to attractor
                plt.plot([self.pos_attractor[0], center_positions[0, ii]],
                         [self.pos_attractor[1], center_positions[1, ii]],
                         '--', color='black')
            else:
                plt.plot([center_positions[0, ind_parent], center_positions[0, ii]],
                         [center_positions[1, ind_parent], center_positions[1, ii]],
                         '--', color='black')
            
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def eval_weights(self):
        pass
