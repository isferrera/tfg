# -*- coding: utf-8 -*-

import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import json
import librosa
import multiprocessing
import argparse
import math
from pylab import *
from PIL import Image

def preprocess_data(src, dst, src_meta, n_processes=15):

    """
    Calls for distibuted preprocessing of the data.

    Parameters:
    -----------
        src: string
            Path to data directory.
        dst: string
            Path to directory where preprocessed data shall be stored.
        stc_meta: string
            Path to meta_information file.
        n_processes: int
            number of simultaneous processes to use for data preprocessing.
    """

    folders = []

    for folder in os.listdir(src):
        # only process folders
        if not os.path.isdir(os.path.join(src, folder)):
            continue
        folders.append(folder)

    pool=multiprocessing.Pool(processes=n_processes)
    _=pool.map(_preprocess_data, [(os.path.join(src, folder), 
                                          os.path.join(dst, folder), 
                                          src_meta) for folder in sorted(folders)])




def _preprocess_data(src_dst_meta):

    """
    Parameters:
    -----------
        src_dst_meta: tuple of 3 strings
            Tuple (path to data directory, path to destination directory, path
            to meta file)
    """

    src, dst, src_meta = src_dst_meta


    metaData = json.load(open(src_meta))

    # create folder for files
    if not os.path.exists(dst):
        os.makedirs(dst)
    # loop over recordings
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):

        # infer sample info from name
        dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")
        # read data
        data,fs = librosa.load(filepath)
        # resample
        data = librosa.core.resample(y=data.astype(np.float64), orig_sr=fs, target_sr=16000, res_type="scipy")
        
        
        ##### Lyon ##### 
        maximo = 0.000610332593980689
        samples = data.size
        decimation_factor =  math.floor((samples/16000)*285.7)
        from lyon.calc import LyonCalc
        calc = LyonCalc()
        coch = calc.lyon_passive_ear(data, decimation_factor = decimation_factor, step_factor = 0.38)
        x = coch.shape[0]
        ct = coch.T
        ct = (ct*255)/maximo
        from skimage.transform import resize
        resized = resize(ct, (28, 28))
        plt.imshow(resized, cmap='gray')
        plt.clim(0,255)
        plt.axis('off')
        plt.savefig(os.path.join(dst, "Im_{}_{}_{}.jpeg".format(dig, vp, rep)),bbox_inches='tight',pad_inches = 0, dpi=7.75)


    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-src', default= "/data", help="Path to folder containing each participant's data directory.")
    parser.add_argument('--destination', '-dst', default="/preprocessed_data", help="Destination where preprocessed data shall be stored.")
    parser.add_argument('--meta', '-m', default="/data/audioMNIST_meta.txt", help="Path to meta_information json file.")

    args = parser.parse_args()

    # preprocessing
    preprocess_data(src=args.source, dst=args.destination, src_meta=args.meta)
    print("FIN")
    
    