from detectron2.structures import BoxMode
import cv2 as cv
import pandas as pd


class MetaDataClass():

    def __init__(self):
        self.train_df = pd.read_csv("dataset\\train_df.csv")

    def create_annotation(self):
        dataset_dicts = []
        for idx, v in enumerate(self.train_df):

            record = {}
            filename = v[0]
            height, width = cv.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            for i in range(len(v[2])):

                annos = v[2][i]

                x1, y1, w, h = annos[0], annos[1], annos[2], annos[3]

                x2, y2 = x1 + w, y1 + h

                obj = {"bbox": [x1, y1, x2, y2],
                       "bbox_mode": BoxMode.XYXY_ABS,
                       "category_id": 0
                       }

                objs.append(obj)

            record["annotations"] = objs

            dataset_dicts.append(record)

        return dataset_dicts
