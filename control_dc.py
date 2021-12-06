import serial
from time import sleep

ser = serial.Serial('/dev/ttyUSB0', 9600)

def ratio_start():
        print('ratio_start')
        ser.write(b'0')
        ser.flush()

def ratio_left():
        # print('ratio_left - 1')
        ser.write(b'1')
        ser.flush()

def ratio_right():
        # print('ratio_right - 2')
        ser.write(b'2')
        ser.flush()

def ratio_up():
        # print('ratio_up - 3')
        ser.write(b'3')
        ser.flush()

def ratio_down():
        # print('ratio_down - 4')
        ser.write(b'4')
        ser.flush()

def ratio_left_up():
        # print('ratio_left_up  - 5')
        ser.write(b'5')
        ser.flush()

def ratio_left_down():
        # print('ratio_left_down - 6')
        ser.write(b'6')
        ser.flush()

def ratio_right_up():
        # print('ratio_right_up - 7')
        ser.write(b'7')
        ser.flush()

def ratio_right_down():
        print('ratio_right_down - 8')
        ser.write(b'8')
        ser.flush()

def ratio_stop():
        # print('ratio_stop - 9')
        ser.write(b'9')
        ser.flush()

def ratio(command):
        if command == 0:
                ratio_start()
        elif command == 1:
                ratio_left()
        elif command == 2:
                ratio_right()
        elif command == 3:
                ratio_up()
        elif command == 4:
                ratio_down()
        elif command == 5:
                ratio_left_up()
        elif command == 6:
                ratio_left_down()
        elif command == 7:
                ratio_right_up()
        elif command == 8:
                ratio_right_down()
        elif command == 9:
                ratio_stop()
        
# while True:
#         ratio(1)
# for i in range(0,10):
#         print(f'ratio({str(i)}) = ', ratio(i))
# try:
#         while True:
#               #truyen lenh
# except KeyboardInterrupt:
#         print("closing...")
