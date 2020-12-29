import os
from .dataset_rgb import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR,DataLoaderVal_deblur,DataLoaderTrain_fusion,DataLoaderVal_fusion,DataLoaderVal_deblur_fusion

def get_training_data(rgb_dir, img_options):
    #assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_training_data_fusion(rgb_dir, img_options):
    #assert os.path.exists(rgb_dir)
    return DataLoaderTrain_fusion(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_validation_data_fusion(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderVal_fusion(rgb_dir, None)


def get_test_images(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderVal_deblur(rgb_dir, None)

def get_test_images_fusion(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderVal_deblur_fusion(rgb_dir, None)


def get_test_data(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    #assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)


