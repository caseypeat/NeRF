import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse
import os
import open3d as o3d


if __name__ == '__main__':

    dirpath = './data/long'

    pcds = o3d.geometry.PointCloud()

    for file in os.listdir(dirpath):
        filepath = os.path.join(dirpath, file)
        pcd = o3d.io.read_point_cloud(filepath)
        pcds += pcd

    o3d.io.write_point_cloud('./data/long2b.pcd', pcds)