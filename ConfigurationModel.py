from Model import Model

from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

model = Model()


class ConfigModel:
    def __init__(self):
        self.cfg = get_cfg()

        if torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cuda"
        else:
            self.cfg.MODEL.DEVICE = "cpu"

        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/"+model.currentModel()))

        self.cfg.MODEL.WEIGHTS = model.modelWeight()

        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4

    def config(self):
        return self.cfg

    def engine(self):
        return self.cfg.MODEL.DEVICE

    def weightModel(self):
        return self.cfg.MODEL.WEIGHTS

    def thresholdModel(self):
        return self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST
