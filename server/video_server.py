import zmq
import time
import random
import google.protobuf.text_format
from server.DroneMsg_pb2 import MsgDrone
# import threading
import numpy as np


# class video_server():
#     """
#     Publish drone state information to unity client via socket
#     """
#     def __init__(self, drone_id):
#         # self.sim_step = sim_step
#         self.message = MsgDrone()
#         self.id = drone_id
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.PUB)
#         if self.id == 1:
#             self.socket.bind("tcp://*:12345") # Chaser=12345
#         else:
#             self.socket.bind("tcp://*:12346") # Target=12346
#
#     def send_state(self,  sim_step, state_now):
#         self.message.id = self.id
#         self.message.sim_step = sim_step
#         state = state_now.tolist()
#         #print(state_now)
#         #print(state[0:3])
#         self.message.pos[:] = state[0:3]
#         self.message.vel[:] = state[3:6]
#         self.message.att_quat[:] = state[6:10]
#         self.message.att_rate[:] = state[10:-2]
#         data = self.message.SerializeToString()
#         self.socket.send(data)

context = zmq.Context()
socket = context.socket(zmq.PUB)
message = msg.MsgDrone()
message.id = 1
message.sim_step = 1
# message.state = state
message.pos[:] = [10, 0.1, 10]
message.att_quat[:] = [0, 0, 0, 1]
message.pos[:] = [10+0.1*t, 0.1+0.1*t, 10+0.1*t]
t = 0

while True:
    t += 1
    message.sim_step = t

    # message = str(random.uniform(-1.0, 1.0)) + " " + str(random.uniform(-1.0, 1.0)) + " " + str(random.uniform(-1.0, 1.0))
    data = message.SerializeToString()
    socket.send(data)
    # socket.send_string(data)
    print(data)
    time.sleep(1)