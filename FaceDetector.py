from Model import Model
from ConfigurationModel import ConfigModel

import time
import cv2 as cv
from detectron2.engine import DefaultPredictor


model = Model()
cfg_model = ConfigModel()


class Detector:

    # detectron model
    def inference(self, file):

        predictor = DefaultPredictor(cfg=cfg_model.config())
        img = file

        start_time = time.time()

        # prediction
        outputs = predictor(img)

        end_time = time.time()

        # Get the bounding boxes, scores, labels
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        # scores = outputs["instances"].scores.numpy()
        class_labels = outputs["instances"].pred_classes.cpu().numpy()

        class_names = {0: 'Face'}
        # score = 0

        for box, class_label in zip(boxes, class_labels):

            x1, y1, x2, y2 = box
            cv.rectangle(img, (int(x1), int(y1)),
                         (int(x2), int(y2)), (255, 0, 0), 2)

            class_name = class_names[class_label]
            # score = "{:.2f}".format(score)

            label = "{}".format(class_name)
            cv.putText(img, label, (int(x1), int(y1)-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate the processing time
        processing_time = end_time - start_time

        return img
