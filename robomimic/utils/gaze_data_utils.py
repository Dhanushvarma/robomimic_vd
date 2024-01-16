import numpy
import re
#TODO: dhanush, check for correctness

class gaze_data_util:

    def __init__(self, screen_horizontal, screen_vertical):

        self.height = screen_vertical
        self.width = screen_horizontal

        #TODO: For now assuming that the pixels coordinates system is top left corner (0,0)
        # Need to write a transform function which aligns the coordiante system if not the same


    def extract_data(self, input_string):
        # Regular expression pattern to match the format
        pattern = r'(\w+)="([\d\.\-]+)"'

        # Finding all matches in the input string
        matches = re.findall(pattern, input_string)

        # Creating a dictionary from the matches
        # Keys are : [FPOGX, FPOGY, FPOGS, FPOGD, FPOGID, FPOGV, CX, CY, CS]
        return {key: float(value) if '.' in value else int(value) for key, value in matches}



    def gaze_pixels(self, gaze_input):

        gaze_data_dict = self.extract_data(gaze_input)

        output_dict = dict.fromkeys(['pixel_x', 'pixel_y'])

        output_dict['pixel_x'] = self.width * gaze_data_dict['FPOGX']
        output_dict['pixel_y'] = self.height * gaze_data_dict['FPOGY']

        return output_dict, gaze_data_dict
