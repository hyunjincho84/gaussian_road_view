import sys
import os
from tools.train_g_splat import *
from datetime import datetime

def train(submodel_num):
    input_path = './colmap_output'
    output_path = './output'
    # activate_conda()
    for i in range(submodel_num):
        construct_gaussian(f"{input_path}/{i+1}/undistorted/")
        print("**************************************")
    
    dirs = [(d, os.path.getmtime(os.path.join(output_path, d))) for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    dirs.sort(key=lambda x: x[1])

    for i, (dir_name, _) in enumerate(dirs, start=1):
        new_name = str(i)
        os.rename(os.path.join(output_path, dir_name), os.path.join(output_path, new_name))

