#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
from pose_estimation.trt_pose_hand.handpose import HandPoseModel
import logging
import os
import sys
import uuid
from concurrent import futures
from time import sleep
import time
import cv2,statistics
import yaml, grpc,torch,tempfile
import numpy as np
sys.path.append('proto')
#sys.path.append(os.getcwd())
#sys.path.append('detection/yolov5')
import proto.metadata_pb2 as metadata_pb2
import proto.metadata_pb2_grpc as metadata_pb2_grpc
logging.basicConfig(level=logging.INFO)
from tracking.yolov4.load_model import Loadv4Model
from tracking.yolov4.detect import Detect
sys.path.append('detection/yolov5_hub')
from detection.yolov5_hub.load_model import Loadv5Model
from detection.yolov5_hub.detect import Detectv5
from FaceMaskDetection.pytorch_infer import LoadMaskModel,DetectMask
from pose_estimation.trt_pose.tasks.human_pose.bodypose import BodyPoseModel

inf = []
ov_head = []
sum = 0
count = 0
class Inference(metadata_pb2_grpc.InferencerServicer):
    def __init__(self):
        self.handpose = HandPoseModel()
        self.yolov4tiny_model = Loadv4Model().load_model()
        self.detect_obj = Detect(self.yolov4tiny_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bodypose = BodyPoseModel()
        self.yolov5s_model = Loadv5Model().load_model()
        self.detect_yolov5 = Detectv5(self.yolov5s_model)
        self.mask_model = LoadMaskModel().load_model()
        self.detect_mask = DetectMask(self.mask_model)


    def Submit(self, request, context):
        #print("connected...")
        #print(request)
        global sum,count
        start_s = time.time()
        frames = []
        if request.is_cloud_exec:
            #print("Grpc: ",start_s - float(request.batch_id))
            #file_nm = f'/tmp/ramdrive/{uuid.uuid4()}.mp4'
            #file = open(file_nm, 'wb')
            #file.write(request.frame)
            img_array = np.asarray(bytearray(request.frame), dtype=np.uint8)
            frames.append(img_array)
            info = request.batch_id
            process_id=info.split("::")[0]
            model= info.split("::")[1]
        else:
            file_nm = request.batch_id
            print(f"Batch id received {file_nm}")
            process_id=0
            cap = cv2.VideoCapture(file.name)
       # while True:
       #     ret, frame = cap.read()
       #     if not ret:
       #         break
       #     frames.append(frame)
            print(f'{len(frames)}')
        """
        Inferencing code goes here
        """
        #do some warmup
        """
        dummy_tensor = torch.zeros(1,3,640,480).to(self.device)
        for _ in range(50):
            self.detect_obj.detect(dummy_tensor)
        """

        #start = time.time()
        if model == 'v4':
            res = self.detect_obj.detect(frames)  #yolov4 tiny hazard vest tracking
        #elif model == 'v5':
        #    res = self.detect_yolov5.detect(frames)    #yolov5s object detection
        #elif model == 'fm':
        #    res = self.detect_mask.inference(frames)   #mask detection
        #elif model == 'bp':
        #    res = self.bodypose.detect_pose(frames)     #bodypose estimation
        #elif model == 'hp':
        #    res = self.handpose.detect_handpose(frames)     #handpose estimation
        #end_i = time.time()
        #inf.append(end_i-start)
        #print("Process %s inf time: %s" % (process_id,end_i-start))


        end = time.time()
        print("time taken: %s" % (end-start_s))
        #print("Process %s overhead %s" % (process_id,end-start_s - (end_i-start)))
        #ov_head.append(end-start_s - (end_i-start))
        sum += end-start_s
        count += 1
        return metadata_pb2.Ack(message="Processed the chunk")
def start_server():
    global sum,count
    #conf_file = open("conf/config.yaml")
    # is_cloud_device = os.environ['is_cloud_device']
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))
    metadata_pb2_grpc.add_InferencerServicer_to_server(Inference(),
                                                       server)
    server.add_insecure_port('[::]:6001')
    try:
        server.start()
        print("Inferencing Server started... ")
        server.wait_for_termination()
    except Exception:
        print(statistics.mean(inf),statistics.mean(ov_head))
        print("Sum %s, count %s, sum/count %s" % (sum,count,sum/count))
    finally:
        print(statistics.mean(inf),statistics.mean(ov_head))
        print("Sum %s, count %s, sum/count %s" % (sum,count,sum/count))
if __name__ == "__main__":
    start_server()
