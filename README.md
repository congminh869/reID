server: test_video_resnet50_thread.py 
client : client.py 

server's line 288: footage_socket.connect('tcp://192.168.0.101:5555') 192.168.0.101 replace with client's IP

client's line 22  : self.s.connect(('192.168.0.102', 8000)) 192.168.0.101 replace with server's IP

##setup run
https://code.luasoftware.com/tutorials/linux/auto-start-python-script-on-boot-systemd/ 
```bash
sudo nano /etc/systemd/system/reID_copy.service
```
check program running
```bash
cat reID_copy.log
```
[Unit]

After=network.service

Description=REID


[Service]

Type=simple

User=mq

WorkingDirectory=/home/mq/Desktop/reID_copy

ExecStart=/bin/bash /home/mq/Desktop/reID_copy/reID_copy.sh



[Install]

WantedBy=multi-user.target




