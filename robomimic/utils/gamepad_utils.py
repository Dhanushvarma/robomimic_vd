from robomimic.Gamepad.Gamepad import Gamepad, available
from robomimic.Gamepad.Controllers import PS4

import numpy as np
import time


def connect_gamepad():
    if not available():
        print('Please connect your gamepad...')
        while not available():
            time.sleep(1.0)
    gamepad = PS4()
    print('Gamepad connected')
    gamepad.startBackgroundUpdates()
    return gamepad


def get_gamepad_action_robosuite(gamepad):
    assert gamepad.isConnected()

    # NOTE(dhanush) : POSITION INPUT
    action = np.zeros((7,))  # NOTE(dhanush) : TO STORE ACTION IN THIS VAR
    action[0] = 0.5 *  gamepad.axis("LEFT-Y")  # NOTE(dhanush) : CORRESPONDS TO W/S Keyboard
    action[1] = 0.5 * gamepad.axis("LEFT-X")  # NOTE(dhanush) : CORRESPONDS TO A/D Keyboard
    action[2] =  - 0.5 * gamepad.axis("RIGHT-Y")  # NOTE(dhanush) : CORRESPONDS TO VERTICAL MOTION IN Z AXIS
    # NOTE(dhanush) :rotation output is limited in the range [-0.3, 0.3]
    action[3] = 0  # NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT X AXIS
    action[4] = 0  # NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT Y AXIS
    action[5] = 0  # NOTE(dhanush) : NO ROTATION INPUT, CORRESPONDS TO ROTATION ABOUT Z AXIS
    # NOTE(dhanush) : GRIPPPER INPUT
    action[6] = -1 + (2 if gamepad.axis("L2") == 1 else 0 )  # NOTE(dhanush) : HAVE TO HOLD THE L1 KEY FOR CLOSING THE GRIPPER

    # NOTE(dhanush) : BREAK EPISODE - PRESS CIRCLE
    break_episode = True if gamepad.beenPressed("CIRCLE") else False

    # NOTE(dhanush) : MOVEMENT ENABLED - NEED TO PRESS THE R2 KEY HARD
    control_enabled = True if gamepad.axis("R2") == 1 else False

    return action, control_enabled, break_episode



# EXAMPLE USAGE

# if __name__ == "__main__":
#
#     gamepad = connect_gamepad()
#
#     while True:
#         # eventType, index, value = gamepad.getNextEvent()
#         # print(eventType, index, value)
#
#         action, control_enabled, break_episode   = get_gamepad_action_robosuite(gamepad)
#         # print(action)
#         if control_enabled:
#             print(control_enabled)
#         if break_episode:
#             print(break_episode)