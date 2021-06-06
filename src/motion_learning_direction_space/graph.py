"""
The learning.graph includes elements to automatically build graphs of GMM's to
simplify the model creation
"""
# Author: Lukas Huber
# email: hubernikus@gmail.com
# License: MIT

import numpy as np


def GraphGMM(DirectionalGMM):
    """ Creats grpah from input gmm

    The main idea is (somehow to an overcompetivtive faimliy):
    - Direct-successor are friends (Grand-grand-...-parents to grand-grand-...-child)
    - Sibilings (plus cousins and all not successors) are rivals
    
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ellipse factor relative to 
        self._ellipse_axes_factor = 4

        self._end_points = None
        self._graph_root = None

    @property
    def n_gaussians(self):
        return self.gmm.n_gaussians

    def ellipse_axes_length(self, it, axes_factor=3):
        """ """
        # Get intersection with circle and then ellipse hull
        if gmm.covariance_type == 'full':
            covariances = self.dpgmm.covariances_[n][:self.dim_space, :][:, :self.dim_space]
        else:
            raise TypeError("Not implemented for unfull covariances")
        covariances = covariances * self._ellipse_axes_factor**self.dim_space
        return covariances

    def get_end_point(self, it):
        """ Return the end point of a specific gaussian parameter.

        Note: this is (simplified) since we assume the orienation being the
        same as it is at the center. """
        
        mean = self.gmm.means_[it, :]

        mean_pos = mean[:self.dim_space]
        mean_dir = mean[-(self.dim_space-1):]

        null_direction = self.null_ds(mean_pos)

        velocity = get_angle_space_inverse(
            dir_angle_space=mean_dir, null_direction=null_direction)

        
        end_point = covariances.dot(velocity)
        return end_point

    def get_rival_weight(self, ):
        """ """
        end_point = np.arange(3)
        pass

    def get_parent(self, it):
        """ Returns parent (int) of the Node [it] as input (int) """
        return self._parent_array[it]

    def get_children(self, it):
        return self._children_list[it]

    def set_parent(self, it, parent):
        self._parent_array[it] = parent
    
    def add_children(self, it, children):
        if isinstance(children, int):
            children = [children]
        self._children_list[it] = self._children_list[it] + children

    def _get_assigned_graph(self):
        pass
    
    def create_graph(self, ):
        """ Create graph from learned GMM."""
        self._end_points = np.zeros((self.dim_space, self.n_gaussians))
        self._parent_array = np.zeros((self.n_gaussians))
        self._children_list = [[] for ii in range(self.n_gaussians)]

        # First 'level' is based on closest to origin
        for ii in range(self.n_gaussians):
            self._end_points[:, ii] = self.get_end_point(it=ii)

        # Get root / main-parent (this could be replaced by optimization / root finding)
        dist_atrractors = np.linalg.norm(
            self._end_points-np.tile(self.attractor, (1, self.n_gaussians)), axis=0)
        self._graph_root = np.argmin(dist_atrractors)
        
        weights = get_mixing_weights(X=self._end_points, weight_factor=self.dim_space)

        weights_without_self = np.copy(weights)
        for ii in range(self.n_gaussians):
            weights_without_self[ii] = 0

        parents_preference = np.argsort(weights_without_self, axis=1)

        # Last one is self, i.e. don't check it
        for ii 
        for it_pref in range(self.n_gaussians-1):
            assigned_list = self.get_assigned_list()
            
            ind_root = (parents_preference[:, it_pref] == self._graph_root)
            
            # Check if any parent is equal to the root
            if any(ind_root):
                # If ii > 0 it means that the prefered parent choice would be another one
                # in that case the prefered graph might be a different one (!)
                # Hence it will be double checked that this is actually the optimal one!
                if it_pref > 0:
                    if np.sum(ind_root) > 1:
                        ind_child = np.argmin(weights_without_self[:, it_pref])

                        self.set_parent(ind_child, parent=ind_root)
                    
                for it_graph in range(np.sum(ind_root)):
                    self.set_parent = parents_preference[jj, ii]
          
                        # raise NotImplementedError("Do it now")
                # Find closest one

    def eval_weights(self):
        pass
