from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
import os
import joblib
import numpy as np
import time


def train_epc(arglist):
    proxy_train({"arglist": arglist})

if __name__ == "__main__":
    arglist = parse_args()
    train_epc(arglist)
