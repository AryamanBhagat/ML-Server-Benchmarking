import os
import sys
import time
import csv
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn

#FILE = Path(__file__).resolve()
#ROOT = FILE.parents[0]  # YOLOv5 root directory
#if str(ROOT) not in sys.path:
 #   sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import csv
#from detection.yolov5.models.common import DetectMultiBackend
#from detection.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
#from detection.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
#                                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
#from detection.yolov5.utils.plots import Annotator, colors, save_one_box
#from load_model import Loadv5Model


class Detectv5(object):
    def __init__(self,model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect(self,batch_frame):    
        #dummy_data = torch.zeros(len(batch_frame),3,640,640).to(self.device)
        #print(batch_frame.shape)
        starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        outs = self.model(batch_frame,size=640)
        ender.record()
        torch.cuda.synchronize()
        inference_time = starter.elapsed_time(ender)
        print("inference time of yolov5s model:",inference_time/1000)
        return outs

#yolov5_model = Loadv5Model().load_model()
#detect = Detectv5(yolov5_model)
#for i in range(10):
    #detect.detect([1])




