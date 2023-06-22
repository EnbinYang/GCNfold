"""
    File to load dataset based on user control from main file
"""
from data.RNAGraph import RNADataset


def LoadData(base_dir, DATASET_NAME, config):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """

    return RNADataset(base_dir, DATASET_NAME, config)
    