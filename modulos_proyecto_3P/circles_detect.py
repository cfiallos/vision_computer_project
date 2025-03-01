import cv2
import numpy as np

class CircleDetector:
    def __init__(self):
        self.__circle_detect = []

    def detect_circles(self, segmented_mask, dp=1.2, minDist=30, param1=50, param2=40, minRadius=10, maxRadius=100):

        if len(segmented_mask.shape) == 3:
            segmented_mask = cv2.cvtColor(segmented_mask, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.uint8)
        segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_CLOSE, kernel)  
        segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_OPEN, kernel)   

        blurred = cv2.GaussianBlur(segmented_mask, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.__circle_detect = self.__filtrar_circulos(circles[0])

        return self.__circle_detect

    def __filtrar_circulos(self, circles, umbral_distancia=20):
      
        circulos_filtrados = []

        for c in circles:
            x, y, r = c
            agregar = True

            for cf in circulos_filtrados:
                x_cf, y_cf, r_cf = cf
                distancia = np.sqrt((x - x_cf) ** 2 + (y - y_cf) ** 2)
                if distancia < umbral_distancia and abs(r - r_cf) < umbral_distancia:
                    agregar = False
                    break

            if agregar:
                circulos_filtrados.append((x, y, r))

        return circulos_filtrados

    def draw_detected_circles(self, frame, segmented_mask):
       
        detected_circles = self.detect_circles(segmented_mask)

        if detected_circles:
            output = frame.copy()  
            for x, y, r in detected_circles:
                cv2.circle(output, (x, y), r, (0, 0, 255), 2)  
                cv2.circle(output, (x, y), 3, (255, 0, 0), -1)  

            return output

        return frame 
    
    
    

        
    