import os
from detectron2 import model_zoo


class Model:
    def __init__(self):

        # set model and test set
        self.model = 'retinanet_R_101_FPN_3x.yaml'

        self.mode_weight = "result\\result-retinaNet101_batchSize=4_itter=1000_lr=0.001\\model_final.pth"

    def currentModel(self):
        return self.model

    def modelWeight(self):
        return self.mode_weight
