import numpy as np

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([   [532.07177734,   0 ,   646.42419434],
                                            [ 0,       532.07177734 ,    360.59887695],
                                            [ 0,                0,     1.0000]
                                        ])
        # 右相机内参
        self.cam_matrix_right = np.array([  [532.071777,   0,   646.424194],
                                            [       0,  532.071777 , 360.598877 ],
                                            [       0,        0,    1.0000]
                                        ])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0,  0, 0,  0,  0]])
        self.distortion_r = np.array([[0,  0 , 0,0 , 0]])

        # 旋转矩阵
        self.R = np.array([ [1.0000,   0.0028,  -0.0050],
                            [-0.0028,  1.0000,   -0.0017],
                            [ 0.0050,    0.0017 ,   1.0000]
                            ])
        # 平移矩阵
        self.T = np.array([[119.79370], [0], [0]])
        # 焦距
        self.focal_length = 532.235962 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
        # 基线距离
        self.baseline = 119.7937 # 单位：mm， 为平移向量的第一个参数（取绝对值）