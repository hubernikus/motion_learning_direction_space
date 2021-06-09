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
        self.ellipse_factor = 2

    @property
    def n_gaussians(self):
        return self.gmm.n_gaussians

    def ellipse_axes_length(self, it):
        """ """
        covariance_matrices = self.dpgmm.covariances_[:, feat_in, :][:, :, feat_in]
        

    def get_end_point(self, it):
        """ Return the end point of a specific gaussian parameter.

        Note: this is (simplified) since we assume the orienation being the
        same as it is at the center. """
        
        mean = self.gmm.means_[it, feat_out]

        mean_pos = mean[:self.dim_space]
        mean_dir = mean[-(self.dim_space-1):]

        null_direction = self.null_ds(mean_pos)

        velocity = get_angle_space_inverse(
            dir_angle_space=mean_dir, null_direction=null_direction)

        # Get intersection with circle and then ellipse hull
        covariance = 
        
        
        return 0

    def get_rival_weight(self, ):
        """ """
        
        pass
    
    def create(self, ):
        """ """

        pass

    def eval_weights(self):
        pass
