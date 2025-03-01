import cv2
import PySimpleGUI as sg
from modulos_proyecto_3P.segmentations import SegmentationOpenCV
from modulos_proyecto_3P.color_range import ColorRange
from modulos_proyecto_3P.interface import create_interface
from modulos_proyecto_3P.model_detect import ModelDetector


class VideoProcessor:
    def __init__(self):
        self.__window = create_interface()
        self.__cap = None
        self.__segmentation = SegmentationOpenCV()
        self.__colorange = ColorRange()
        self.model_detector = ModelDetector("cnn_model.pth")
        
        while True:
            event, values = self.__window.read(timeout=20)
            
            if event in (sg.WIN_CLOSED, 'Salir'):
                break

            if event == 'Iniciar' and self.__cap is None:
                self.__cap = cv2.VideoCapture(0)

            if event == 'Detener' and self.__cap is not None:
                self.__cap.release()
                self.__cap = None
                cv2.destroyAllWindows()

            if self.__cap is not None and self.__cap.isOpened():
                ret, frame = self.__cap.read()
                if not ret:
                    break
                self.__segmentation.set_frame(frame)         

                if values['SEGMHSV']:
                    color_space = 'HSV'
                    color_name = None
                    for color in ['ZENCHUA', 'ZUTEM', 'ZUCCHINI', 'ZODOBA', 'ZEPPELIN', 'ZULU', 'DEFAULT']:
                        if values[color]:
                            color_name = color
                            break
                    
                    if color_name:
                        frame = self.__segmentation.apply_color_segmentation(color_space=color_space, color_name=color_name)
                        low, high = self.__colorange.get_color_ranges(color_name)
                        if low is not None and high is not None:
                            self.__window['LOW_H'].update(value=low[0])
                            self.__window['LOW_S'].update(value=low[1])
                            self.__window['LOW_V'].update(value=low[2])
                            self.__window['HIGH_H'].update(value=high[0])
                            self.__window['HIGH_S'].update(value=high[1])
                            self.__window['HIGH_V'].update(value=high[2])

                    low_hsv = (values['LOW_H'], values['LOW_S'], values['LOW_V'])
                    high_hsv = (values['HIGH_H'], values['HIGH_S'], values['HIGH_V'])
                    self.__segmentation.set_color_ranges(*low_hsv, *high_hsv)                    
                    frame = self.__segmentation.apply_color_segmentation(color_space=color_space, color_name=color_name)

                if values['SEGMKMEANS']:
                    k = int(values['KMEANS_K'])
                    frame = self.__segmentation.apply_kmeans_segmentation(k)
  
                if values['DECTCONT']:
                    frame = self.__segmentation.apply_border_detector()

                if values.get('DRAWCIRC', False):
                    frame = self.__segmentation.draw_detected_circles()

                if values['CNNMODEL']:
                    resultado_prediccion = self.model_detector.predict(frame)
                    cv2.putText(frame, f"Prediccion: {resultado_prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.imshow('Video', frame)


        if self.__cap is not None:
            self.__cap.release()
        cv2.destroyAllWindows()
        self.__window.close()

if __name__ == '__main__':
 video_processor = VideoProcessor()
