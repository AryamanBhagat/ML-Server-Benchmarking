import logging
import os
import sys
import uuid
from concurrent import futures
from time import sleep
import time
import cv2
import yaml, grpc

sys.path.append(os.getcwd())
sys.path.append('proto')
#sys.path.append('tracking')
#sys.path.append('tracking/yolov4/')
#sys.path.append('FaceMaskDetection/')

import proto.metadata_pb2 as metadata_pb2
import proto.metadata_pb2_grpc as metadata_pb2_grpc

logging.basicConfig(level=logging.INFO)
import torch

from tracking.yolov4.load_model import Loadv4Model
from tracking.yolov4.detect import Detectv4
#from detection.yolov5.load_model import Loadv5Model
#from detection.yolov5.detect import Detectv5
#from FaceMaskDetection import pytorch_infer


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Inference(metadata_pb2_grpc.InferencerServicer):
    def __init__(self):
        self.yolov4tiny_model = Loadv4Model().load_model()
        #self.yolov5s_model = Loadv5Model().load_model()
        #self.facemask_model = pytorch_infer.load_model()

    def Submit(self, request, context):
        #print("connected...")
        print(request)
        start = time.time()
        if request.is_cloud_exec:
            file_nm = f'{uuid.uuid4()}.mp4'
            file = open(file_nm, 'wb')
            file.write(request.frame)
        else:
            file_nm = request.batch_id
            print(f"Batch id received {file_nm}")

        frames = []
        cap = cv2.VideoCapture(file_nm)
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frames.append(frame)
        print(f'{len(frames)}')
        """
        Inferencing code goes here
        """
        start = time.time()
        res = Detect(self.yolov4tiny_model).detect(frames)  #yolov4 tiny hazard vest tracking
        #res = pytorch_infer.inference(self.facemask_model,frames)
        #print("inf time:", time.time()-start)
        #_ = v5Detect(self.yolov5s_model).detect(frames) #yolov5s object detection

        ########################################
        #sleep(10)
        end = time.time()
        print("time taken:",end-start)
        return metadata_pb2.Ack(message="Processed the chunk")


def start_server():
    #conf_file = open("conf/config.yaml")
    # is_cloud_device = os.environ['is_cloud_device']
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    metadata_pb2_grpc.add_InferencerServicer_to_server(Inference(),
                                                       server)
    server.add_insecure_port('[::]:6001')
    server.start()
    print("Inferencing Server started... ")
    server.wait_for_termination()


if __name__ == "__main__":
    start_server()
