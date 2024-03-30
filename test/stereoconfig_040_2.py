import numpy as np

# Camera Parameters 
class stereoCamera(object):
    def __init__(self):
        # Left Camera Matrix
        self.cam_matrix_left = np.array([   [830.5873,   -3.0662,  658.1007],
                                            [       0,  830.8116,  482.9859],
                                            [       0,         0,         1]
                                        ])
        # Right Camera Matrix
        self.cam_matrix_right = np.array([  [830.4255,   -3.5852,  636.8418],
                                            [       0,  830.7571,  476.0664],
                                            [       0,         0,         1]
                                        ])

        # Camera Distortion matrix:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0806, 0.3806, -0.0033, 0.0005148, -0.5229]])
        self.distortion_r = np.array([[-0.0485, 0.2200, -0.002,  0.0017,    -0.2876]])

        # Rotation Matrix
        self.R = np.array([ [      1,  0.0017, -0.0093],
                            [-0.0018,  1.0000, -0.0019],
                            [ 0.0093,  0.0019,  1.0000]   
                            ])

        # Translation Matrix
        self.T = np.array([[-119.9578], [0.1121], [-0.2134]])

        # Facal length 
        self.focal_length = 859.367 # Default，Generally, Q[2,3] in the reprojection matrix Q after stereoscopic correction is taken.

        # Baseline distance
        self.baseline = 119.9578 # in mm， First parameter of translation vector（ABS value）

        


