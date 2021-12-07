import argparse
import cv2
import numpy as np
import zmq
import socket
import pickle
import time
from random import randint
# from constants import PORT
from utilss import string_to_image

#PORT = '8888'

IP_CONNECT_SERVER = '192.168.6.52'

def receive_stream(display=True):
    print('start--------------------init')
    
    context = zmq.Context()
    footage_socket = context.socket(zmq.SUB)
    footage_socket.bind('tcp://*:5555')
    footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_CONNECT_SERVER , 8000))

    display = True
    current_frame = None
    keep_running = True

    print('finish----------------------init')

    keep_running = True
    print('1')
    while footage_socket and keep_running:
        try:
            frame = footage_socket.recv_string()
            current_frame = string_to_image(frame)
            high, weight = current_frame.shape[:2]
            bboxes = []
            colors = []


            if display:
                print('check display')
                while True:
                    print('check frame 000000000000000000000')
                    bbox = cv2.selectROI('MultiTracker', current_frame)
                    bboxes.append(bbox)
                    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                    print("Press q to quit selecting boxes and start tracking")
                    print("Press any other key to select next object")
                    k = cv2.waitKey(0) & 0xFF
                    if (k == 113):  # q is pressed
                        break
                        print('breal----------------')
                    print(bbox)
                print('select box done -----------')

                while True:
                    data_rec = s.recv(1024)
                    if data_rec == b'chon anh':
                        print('oke')
                        print('bboxes', bboxes)
                        data=pickle.dumps(bboxes)
                        s.sendall(data)
                        display =False

                    elif data_rec.decode() == 'close':
                        print('---------------elif---------------close')
                        cv2.destroyAllWindows()
                        break
                    else:
                        print('else')
                        break
            else:
                cv2.imshow("Stream", current_frame)
                p = cv2.waitKey(1)
                if (p == 113):  # q is pressed
                    s.sendall('close'.encode())
                    print('---------------gui xong close----')
                    time.sleep(2)
                    break  
        except KeyboardInterrupt:

            cv2.destroyAllWindows()
            break
    print("Streaming Stopped!")


if __name__ == '__main__':
    receive_stream()
