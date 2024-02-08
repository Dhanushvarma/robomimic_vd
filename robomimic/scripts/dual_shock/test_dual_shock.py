import sys
sys.path.insert(0, "/home/dhanush/Gamepad")
from Gamepad import *
import numpy as np


class PS4(Gamepad):
    fullName = 'PlayStation 4 controller'

    def __init__(self, joystickNumber = 0):
        Gamepad.__init__(self, joystickNumber)
        self.axisNames = {
            0: 'LEFT-X',
            1: 'LEFT-Y',
            2: 'L2',
            3: 'RIGHT-X',
            4: 'RIGHT-Y',
            5: 'R2',
            6: 'DPAD-X',
            7: 'DPAD-Y'
        }
        self.buttonNames = {
            0:  'CROSS',
            1:  'CIRCLE',
            2:  'TRIANGLE',
            3:  'SQUARE',
            4:  'L1',
            5:  'R1',
            6:  'L2',
            7:  'R2',
            8:  'SHARE',
            9:  'OPTIONS',
            10: 'PS',
            11: 'L3',
            12: 'R3'
        }
        self._setupReverseMaps()

import time

# def available(joystickNumber = 0):
#     """Check if a joystick is connected and ready to use."""
#     joystickPath = '/dev/input/js' + str(joystickNumber)
#     return os.path.exists(joystickPath)


def connect_gamepad():
    if not available():
        print('Please connect your gamepad...')
        while not available():
            time.sleep(1.0)
    gamepad = PS4()
    print('Gamepad connected')
    gamepad.startBackgroundUpdates()
    return gamepad

def get_gamepad_action(gamepad):
    assert gamepad.isConnected()

    take_control = gamepad.isPressed("L2")
    act_discrete_update = -1 if gamepad.beenPressed("CROSS") else 1
    act_new = 0.1 * np.array([gamepad.axis("LEFT-X"), -gamepad.axis("LEFT-Y"),
                              -float(gamepad.isPressed("L1")) + float(gamepad.isPressed("R1"))])
    reset = gamepad.beenPressed("SQUARE")
    terminate = gamepad.beenPressed("PS")
    if gamepad.beenPressed("CIRCLE"):
        next_window = True
    else:
        next_window = False
    return take_control, act_new, act_discrete_update, reset, terminate, next_window


def get_gamepad_action_robosuite(gamepad):
    assert gamepad.isConnected()

    # NOTE(dhanush) : POSITION INPUT
    action = np.zeros((7,)) #NOTE(dhanush) : TO STORE ACTION IN THIS VAR
    action[0] = gamepad.axis("LEFT-Y") #NOTE(dhanush) : CORRESPONDS TO W/S Keyboard
    action[1] = gamepad.axis("LEFT-X") #NOTE(dhanush) : CORRESPONDS TO A/D Keyboard
    action[2] = gamepad.axis("RIGHT-Y") #NOTE(dhanush) : CORRESPONDS TO VERTICAL MOTION IN Z AXIS
    # NOTE(dhanush) :rotation output is limited in the range [-0.3, 0.3]
    action[3] = 0 #NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT X AXIS
    action[4] = 0  # NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT Y AXIS
    action[5] = 0  # NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT Z AXIS
    # NOTE(dhanush) : GRIPPPER INPUT
    action[6] = -1 + gamepad.isPressed("L1") # NOTE(dhanush) : HAVE TO HOLD THE L1 KEY FOR CLOSING THE GRIPPER

    # NOTE(dhanush) : BREAK EPISODE - PRESS CIRCLE
    break_episode = True if gamepad.beenPressed("CIRCLE") else False

    # NOTE(dhanush) : MOVEMENT ENABLED - NEED TO PRESS THE R2 KEY HARD
    control_enabled = True if gamepad.axis("R2") == 1 else False

    return action, control_enabled, break_episode



if __name__ == "__main__":

    gamepad = connect_gamepad()

    while True:
        # eventType, index, value = gamepad.getNextEvent()
        # print(eventType, index, value)

        action, control_enabled, break_episode   = get_gamepad_action_robosuite(gamepad)
        # print(action)
        if control_enabled:
            print(control_enabled)
        if break_episode:
            print(break_episode)




# def connect_gamepad():
#     if not Gamepad.available():
#         print('Please connect your gamepad...')
#         while not Gamepad.available():
#             time.sleep(1.0)
#     gamepad = Gamepad.PS4()
#     print('Gamepad connected')
#     gamepad.startBackgroundUpdates()
#     return gamepad






# gamepad = connect_gamepad()
# gamepad_object = connect_gamepad()

# for i in range(100):

    # _, action, _, _, _, _ = get_gamepad_action(gamepad_object)


#!/usr/bin/env python
# coding: utf-8

# # Load the gamepad and time libraries
# import Gamepad
# import time
#
# # Make our own custom gamepad
# # The numbers can be figured out by running the Gamepad script:
# # ./Gamepad.py
# # Press ENTER without typing a name to get raw numbers for each
# # button press or axis movement, press CTRL+C when done
# class CustomGamepad(Gamepad.Gamepad):
#     def __init__(self, joystickNumber = 0):
#         Gamepad.Gamepad.__init__(self, joystickNumber)
#         self.axisNames = {
#             0: 'LEFT-X',
#             1: 'LEFT-Y',
#             2: 'RIGHT-Y',
#             3: 'RIGHT-X',
#             4: 'DPAD-X',
#             5: 'DPAD-Y'
#         }
#         self.buttonNames = {
#             0:  '1',
#             1:  '2',
#             2:  '3',
#             3:  '4',
#             4:  'L1',
#             5:  'L2',
#             6:  'R1',
#             7:  'R2',
#             8:  'SELECT',
#             9:  'START',
#             10: 'L3',
#             11: 'R3'
#         }
#         self._setupReverseMaps()
#
# # Gamepad settings
# gamepadType = CustomGamepad
# buttonHappy = '3'
# buttonBeep = 'L3'
# buttonExit = 'START'
# joystickSpeed = 'LEFT-Y'
# joystickSteering = 'RIGHT-X'
#
# # Wait for a connection
# if not Gamepad.available():
#     print('Please connect your gamepad...')
#     while not Gamepad.available():
#         time.sleep(1.0)
# gamepad = gamepadType()
# print('Gamepad connected')
#
# # Set some initial state
# speed = 0.0
# steering = 0.0
#
# # Handle joystick updates one at a time
# while gamepad.isConnected():
#     # Wait for the next event
#     eventType, control, value = gamepad.getNextEvent()
#
#     # Determine the type
#     if eventType == 'BUTTON':
#         # Button changed
#         if control == buttonHappy:
#             # Happy button (event on press and release)
#             if value:
#                 print(':)')
#             else:
#                 print(':(')
#         elif control == buttonBeep:
#             # Beep button (event on press)
#             if value:
#                 print('BEEP')
#         elif control == buttonExit:
#             # Exit button (event on press)
#             if value:
#                 print('EXIT')
#                 break
#     elif eventType == 'AXIS':
#         # Joystick changed
#         if control == joystickSpeed:
#             # Speed control (inverted)
#             speed = -value
#         elif control == joystickSteering:
#             # Steering control (not inverted)
#             steering = value
#         print('%+.1f %% speed, %+.1f %% steering' % (speed * 100, steering * 100))
