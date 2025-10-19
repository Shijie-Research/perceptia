# Copyright (c) Perceptia Contributors. All rights reserved.
import faster_coco_eval
import torchvision


faster_coco_eval.init_as_pycocotools()


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
