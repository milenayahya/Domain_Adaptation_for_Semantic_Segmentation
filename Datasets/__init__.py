import os
DATASETS_BASE_PATH = os.path.abspath("E:\\Poli\\AML\\Datasets")
PROJECT_BASE_PATH = os.path.abspath("E:\\Poli\\AML\\Domain_Adaptation_for_Semantic_Segmentation")
CITYSCAPES_BASE_PATH = os.path.join(DATASETS_BASE_PATH, "Cityspaces")
CITYSCAPES_CROP_SIZE = (512, 1024)

GTA5_BASE_PATH = os.path.join(DATASETS_BASE_PATH, "GTA5")
GTA5_CROP_SIZE = (720, 1280)