import cv2
import numpy as np

class ColorRange:
    def __init__(self):
        self.color_low_range = None
        self.color_high_range = None

    def set_color_ranges(self, LOW_H, LOW_S, LOW_V, HIGH_H, HIGH_S, HIGH_V):
        self.color_low_range = np.array([LOW_H, LOW_S, LOW_V], dtype=np.uint8)
        self.color_high_range = np.array([HIGH_H, HIGH_S, HIGH_V], dtype=np.uint8)
        
    def get_color_ranges(self, color_name):
        color_ranges = {
            'ZENCHUA': ((154, 60, 0), (247, 255, 255)),
            'ZUTEM': ((0, 154, 99), (34, 208, 255)),  
            'ZUCCHINI': ((34, 30, 0), (85, 255, 255)),  
            'ZODOBA': ((0, 27, 83), (23, 119, 110)),  
            'ZEPPELIN': ((63, 0, 0), (255, 146, 50)),  
            'ZULU': ((91, 0, 90), (177, 55, 119)),
            'DEFAULT': ((0, 0, 0), (255, 255, 255)),
        }
        return color_ranges.get(color_name, (None, None))