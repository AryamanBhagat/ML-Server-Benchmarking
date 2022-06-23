import json
import os
import sys
import time

import grpc

sys.path.append('proto')

import proto.metadata_pb2_grpc as m_grpc
import proto.metadata_pb2 as m_messages

"""
sample metadata file
{
drone_id:<>,
batch_size:<>,
batch_duration:<>,
batch_details:
{<video_id>:{
    start_time:<>,
    end_time:<>,}
}
}
"""
# metadata_json = json.load(open('video_chunks/metadata.json'))
# drone_id = metadata_json["drone_id"]
# batch_size = metadata_json["batch_size"]
# batch_duration = metadata_json["batch_duration"]

FOG_IP = "127.0.0.1" 
FOG_GPRC_PORT = "6001"
metadata = open('metadata_1frame')
path_modifier = "/home/dreamlab/Desktop/1frame"

def send_batch_details():
    # create a grpc stub in init
    # form the batch details and send to java process
    channel = grpc.insecure_channel(f"{FOG_IP}:{FOG_GPRC_PORT}", options=(('grpc.enable_http_proxy', 0),))
    stub = m_grpc.InferencerStub(channel)

    for batch_id in metadata.readlines():
        print(batch_id)
        s = time.time()
        resp = stub.Submit(m_messages.JobDetails(batch_id=os.path.join(path_modifier,batch_id.strip()), is_cloud_exec=False))

        print(time.time() - s)
        # print(resp.message)


send_batch_details()

