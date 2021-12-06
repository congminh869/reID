server: test_video_resnet50_thread.py 
client : client.py 

server's line 288: footage_socket.connect('tcp://192.168.0.101:5555') 192.168.0.101 replace with client's IP
client's line 22  : self.s.connect(('192.168.0.102', 8000)) 192.168.0.101 replace with server's IP
