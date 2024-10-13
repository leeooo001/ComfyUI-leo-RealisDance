from annotator.dwpose import DWposeDetector
import math
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', help = 'input path')
    args = parser.parse_args()

    input_path = args.input_path
        
    pose = DWposeDetector()
    outs = []
    for i, image_name in tqdm(enumerate(os.listdir(input_path))):
        test_image = os.path.join(input_path, image_name)
        oriImg = cv2.imread(test_image)  # B,G,R order
        out = pose(oriImg)
        outs.append(out)

    pkl_file = os.path.join(os.path.dirname(input_path), 'dwpose.pkl')
    with open(pkl_file, 'wb') as file:
        pickle.dump(outs, file)

    '''
    pkl_file1 = r"F:\AIGC\Video\2409-RealisDance\main\__assets__\demo_seq\dwpose_1.pkl"
    pkl_file2 = r"F:\AIGC\Video\2409-RealisHuman\main\test\002\dwpose.pkl"

    with open(pkl_file1, 'rb') as file1:
        pkl_data1 = pickle.load(file1)
    with open(pkl_file2, 'rb') as file2:
        pkl_data2 = pickle.load(file2)
    print(pkl_data1[0])
    print(pkl_data2[0])
    '''




