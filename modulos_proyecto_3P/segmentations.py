import cv2
import numpy as np
from modulos_proyecto_3P.circles_detect import CircleDetector
from modulos_proyecto_3P.color_range import ColorRange



class SegmentationOpenCV:
    def __init__(self):
        self.__frame = None
        self.__frame_hsv = None
        self.__mask = None
        self.__color_segmentation = None
        self.__detect_border = None
        self.__colorange = ColorRange()
        self.__circle_detector = CircleDetector()
        self.__k_kmeans: None

    def set_frame(self, frame):
        self.__frame = frame
        self.__frame_hsv = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

    def set_color_ranges(self, LOW_H, LOW_S, LOW_V, HIGH_H, HIGH_S, HIGH_V):
        self.__colorange.set_color_ranges(LOW_H, LOW_S, LOW_V, HIGH_H, HIGH_S, HIGH_V)
        
    def apply_color_segmentation(self, color_space='HSV', color_name = None ):
        if color_space == 'HSV':
            low, high = self.__colorange.get_color_ranges(color_name)
            if low is None or high is None:
                return self.__frame 
            mask = cv2.inRange(self.__frame_hsv, self.__colorange.color_low_range, self.__colorange.color_high_range)
            self.__mask = mask 
            self.__color_segmentation = cv2.bitwise_and(self.__frame, self.__frame, mask=mask)
            return self.__color_segmentation
        
        return self.__color_segmentation
    
    def apply_border_detector(self):
        contours, _ = cv2.findContours(self.__mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__detect_border = cv2.drawContours(self.__color_segmentation, contours, -1, (0, 255, 0), 2)
        return self.__detect_border

    def draw_detected_circles(self):
        """
        Aplica la detección de círculos en la imagen segmentada.
        """
        if self.__mask is None:
            print("Error: No se puede detectar círculos sin una segmentación previa.")
            return self.__frame

        return self.__circle_detector.draw_detected_circles(self.__frame, self.__mask)
        
    def apply_kmeans_segmentation(self, k):
        self.__k_kmeans = k
        if self.__frame is None:
            print("Error: No se ha recibido un frame válido.")
            return None

        Z = self.__frame.reshape((-1, self.__k_kmeans))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(Z, self.__k_kmeans, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        self.__color_segmentation = segmented_image.reshape(self.__frame.shape)

        return self.__color_segmentation