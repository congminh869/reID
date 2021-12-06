#sudo nano /etc/systemd/system/reID_copy.service
cd /home/mq/Desktop/reID_copy/
sudo python3 /home/mq/Desktop/reID_copy/server_multi_thread_reID.py >> /home/mq/Desktop/reID_copy/logs/reID_copy.log 2>&1
