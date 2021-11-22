from gpiozero import Servo
import math
import time

from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory()

servo = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000, pin_factory=factory)

servo.value = math.sin(math.radians(45))

degree_left = degree_right = 0
degree_up = degree_down = 0

g_start = 0
left = 1
right = 2
up = 3
down = 4
left_up = 5
left_down = 6
right_up = 7
right_down = 8
g_stop = 9

g_start_lr = 0
g_start_ud = 0

G_FACTORY = 2
RATIO = float(5/90)
START_DEGREE = 45*RATIO + G_FACTORY
DEGREE_NOW_LR = START_DEGREE
DEGREE_NOW_UD = START_DEGREE

def ratio_start():
    servo.value = math.sin(math.radians(START_DEGREE))

def ratio_left():
    #degree_left++;
    DEGREE_NOW_LR = DEGREE_NOW_LR - 1
    servo.value = math.sin(math.radians(DEGREE_NOW_LR))

def ratio_right():
    #degree_right++;
    DEGREE_NOW_LR = DEGREE_NOW_LR + 1
    servo.value = math.sin(math.radians(DEGREE_NOW_LR))

def ratio_up():
    #degree_up++;
    DEGREE_NOW_UD = DEGREE_NOW_UD + 1
    servo.value = math.sin(math.radians(DEGREE_NOW_UD))

def ratio_down():
    #degree_down++;
    DEGREE_NOW_UD = DEGREE_NOW_UD - 1
    servo.value = math.sin(math.radians(DEGREE_NOW_UD))

def ratio_left_up():
    ratio_left()
    ratio_up()

def ratio_left_down():
    ratio_left()
    ratio_down()

def ratio_right_up():
    ratio_right()
    ratio_up()

def ratio_right_down():
    ratio_right()
    ratio_down()

def ratio_stop():
    #degree_left = degree_right = degree_up = degree_down = 0
    #degree_left_up = degree_left_down = degree_right_up = degree_right_down = 0
    servo.value = None

def ratio(command):
    switcher={
        0:ratio_start,
        1:ratio_left,
        2:ratio_right,
        3:ratio_up,
        4:ratio_down,
        5:ratio_left_up,
        6:ratio_left_down,
        7:ratio_right_up,
        8:ratio_right_down,
        9:ratio_stop
        }
    return switcher.get(command, "Invalid status_ratio")

# while True:
#     ratio(0)
#     while True:
#         ratio(1)
    #for i in range(0, 120):
        #servo.value = math.sin(math.radians(i))
        #time.sleep(0.1)
