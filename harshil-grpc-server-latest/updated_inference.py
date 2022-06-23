from pose_estimation.trt_pose_hand.handpose import HandPoseModel
import json
import logging
import os
import sys
import uuid
from concurrent import futures
from time import sleep
import time
import cv2,statistics
import yaml, grpc,torch,tempfile
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
from collections import defaultdict


inf = []
ov_head = []
e2e = []
sum = 0
count = 0

inf_dict = defaultdict(list)
e2e_dict = defaultdict(list)
logger_inst = {}
def create_logger(logger_name, filename='default'):
    if logger_name in logger_inst:
        return logger_inst[logger_name]
    LOG = logging.getLogger(f"{logger_name}")

    file_fmt = logging.Formatter(fmt='%(message)s')
    console_fmt = logging.Formatter(fmt='%(message)s')

    file_handlr = logging.FileHandler(f"{filename}.log")
    console_handlr = logging.StreamHandler(stream=sys.stdout)

    file_handlr.setFormatter(file_fmt)
    console_handlr.setFormatter(console_fmt)

    LOG.addHandler(file_handlr)
    LOG.addHandler(console_handlr)

    LOG.setLevel("INFO")
    logger_inst[logger_name] = LOG
    return LOG

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
        self.logger = create_logger(sys.argv[1],sys.argv[1])


    def Submit(self, request, context):
        #print("connected...")
        #print(request)
        global sum,count
        start_s = time.time()
        #with tempfile.NamedTemporaryFile(dir='/tmp/ramdrive') as file:
        if request.is_cloud_exec:
            #print("Grpc: ",start_s - float(request.batch_id))
            file_nm = f'/tmp/ramdrive/{uuid.uuid4()}.mp4'
            file = open(file_nm, 'wb')
            file.write(request.frame)
            model= request.dnn_model
            task_id = request.task_id
        else:
            file_nm = request.batch_id
            print(f"Batch id received {file_nm}")
            process_id=0
            task_id = request.task_id
            model= request.dnn_model
        frames = []
        cap = cv2.VideoCapture(file_nm)
        r_s = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        self.logger.info(f'read time {time.time() - r_s}. Frames {len(frames)}')
       # print(f'{len(frames)}')
        """
        Inferencing code goes here
        """
        #do some warmup
        """
        dummy_tensor = torch.zeros(1,3,640,480).to(self.device)
        for _ in range(50):
            self.detect_obj.detect(dummy_tensor)
        """

        start = time.time()
        if model == 'HAZARD_VEST':
            #pass
            res = self.detect_obj.detect(frames)  #yolov4 tiny hazard vest tracking
        elif model == 'CROWD_DENSITY':
            #pass
            res = self.detect_yolov5.detect(frames)    #yolov5s object detection
        elif model == 'MASK_DETECTION':
            #pass
            res = self.detect_mask.inference(frames)   #mask detection
        elif model == 'BODY_POSE_ESTIMATION':
            #pass
            res = self.bodypose.detect_pose(frames)     #bodypose estimation
        elif model == 'HAND_POSE_ESTIMATION':
            res = self.handpose.detect_handpose(frames)     #handpose estimation
        elif model == 'DISTANCE_ESTIMATION':
            #pass
            res = self.detect_obj.detect(frames,dist=True)
        end_i = time.time()
        #inf.append(end_i-start)
        self.logger.info("task_id %s model %s  inf time: %s" % (task_id,model,end_i-start))

        ########################################
        #sleep(10)
        end = time.time()
        self.logger.info("task_id %s model %s time taken: %s" % (task_id,model ,end-start_s))
        #print("Process %s overhead %s" % (process_id,end-start_s - (end_i-start)))
        #ov_head.append(end-start_s - (end_i-start))
        sum += end-start_s
        count += 1


        inf_dict[0].append(end_i-start)
        e2e_dict[0].append(end-start_s)

        return metadata_pb2.Ack(message="Processed the chunk",task_id=task_id)
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
        #print("Sum %s, count %s, sum/count %s" % (sum,count,sum/count))
    finally:
        #print("Sum %s, count %s, sum/count %s" % (sum,count,sum/count))
        print("Mean inf %s, Median Inf %s" % (statistics.mean(inf_dict[0]), statistics.median(inf_dict[0])))
        #print(statistics.mean(inf_dict[0]),statistics.meanr(ov_head))
        with open("inf_result_yolov4.json", "w") as json_file:
            json.dump(inf_dict, json_file)
        with open("e2e_result_yolov4.json", "w") as json_file:
            json.dump(e2e_dict, json_file)

def load_yolov4():
    import torcheia

    model  = torch.jit.load('/home/ubuntu/cv_tasks/tracking/yolov4/best_overall.ts.pt', map_location=torch.device('cpu'))
    torch._C._jit_set_profiling_executor(False)
    eia_model = torcheia.jit.attach_eia(model, 0)
    return eia_model
if __name__ == "__main__":
    start_server()
