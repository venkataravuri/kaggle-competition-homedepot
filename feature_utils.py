"""
__file__
    feature_utils.py
__description__
    Adopted from @Chenglong Chen's code < https://github.com/ChenglongChen/Kaggle_CrowdFlower >
    This file provides utils for generating features.
__author__
    Venkata Ravuri < venkat@nikhu.com >
"""


def try_divide(x, y, val=0.0):
    """
    	Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def dump_feat_name(feat_names, feat_name_file):
    """
        save feat_names to feat_name_file
    """
    with open(feat_name_file, "w") as f:
        for i, feat_name in enumerate(feat_names):
            # f.write("('%s', SimpleTransform()),\n" % feat_name)
            f.write("%s\n" % feat_name)
