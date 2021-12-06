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
class StreamViewer:
    def __init__(self, port):
        print(port)
        print('start--------------------init')
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:5555')
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('192.168.0.102', 8000))
        self.display = True
        self.current_frame = None
        self.keep_running = True
        print('finish----------------------init')

    def receive_stream(self, display=True):
        self.keep_running = True
        print('1')
        while self.footage_socket and self.keep_running:
            # print(self.footage_socket)
            try:
                # print('try')
                # print('2')
                frame = self.footage_socket.recv_string()
                # print('1')
                # print('fname', frame)
                self.current_frame = string_to_image(frame)
                high, weight = self.current_frame.shape[:2]
                # print('********************')
                # print(high, weight)
                # print('********************')
                # self.current_frame = cv2.resize(self.current_frame, (high/2,weight/2))

                bboxes = []
                colors = []


                if self.display:
                    print('check display')
                    while True:
                        # cv2.imshow("Stream", self.current_frame)
                        print('check frame 000000000000000000000')
                        bbox = cv2.selectROI('MultiTracker', self.current_frame)
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

                        #socket
                    while True:
                        data_rec = self.s.recv(1024)
                        if data_rec == b'chon anh':
                            print('oke')
                            data=pickle.dumps(bboxes)
                            self.s.sendall(data)
                            self.display =False
                            # break
                            # data_recv = s.recv(1024)
                            # data = pickle.loads(data_recv)
                        elif data_rec.decode() == 'close':
                            print('---------------elif---------------close')
                            # data=pickle.dumps([1])
                            # self.s.sendall('close'.encode())
                            # self.s.close()
                            cv2.destroyAllWindows()
                            break
                        else:
                            print('else')
                            break
                            # self.s.sendall('close'.encode())
                else:
                    cv2.imshow("Stream", self.current_frame)
                    # self.s.close()
                    p = cv2.waitKey(1)
                    if (p == 113):  # q is pressed
                        self.s.sendall('close'.encode())
                        print('---------------gui xong close----')
                        time.sleep(2)
                        break  
                # print('done-----------------------------')
            except KeyboardInterrupt:

                cv2.destroyAllWindows()
                break
            # print('-------------------', self.keep_running)
        # self.s.close()
        print("Streaming Stopped!")

    def stop(self):

        self.keep_running = False

def main():
    # port = PORT

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Viewer to use, default'
                             ' is ', required=False)

    args = parser.parse_args()
    if args.port:
        port = 5555#args.port
    port = 5556
    print('------------------------------------------------------------')

    stream_viewer = StreamViewer(port)
    print('------------------------------------------------------------')
    stream_viewer.receive_stream()


if __name__ == '__main__':
    main()

