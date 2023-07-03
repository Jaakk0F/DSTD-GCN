# encoding: utf-8
"""
@author:  Jiajun Fu
@contact: Jaakk0F@foxmail.com
"""

from .dstdgcn import DSTDGCN  # For qualtitative results
# from .dstdgcn_fast import DSTDGCN  # For speed


def get_model(model_type, **model_opts):
    return {
        "dstdgcn": DSTDGCN,
    }[model_type](**model_opts[model_type])
