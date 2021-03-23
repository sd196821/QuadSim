import zmq
import time
import random
import google.protobuf.text_format
import DroneMsg_pb2 as msg
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:12345")

message = msg.MsgDrone()
message.id = 1
message.sim_step = 1

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