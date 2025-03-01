import cv2
import torch
import torchvision.transforms as transforms
from modulos_proyecto_3P.model_trained import CNN  

class ModelDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, frame):
        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        imagen_t = self.transform(imagen_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            salida = self.model(imagen_t)
            prediccion = torch.argmax(salida, dim=1).item()

        clases = ["Avion", "Automovil", "Pajaro", "Gato", "Venado", "Perro", "Rana", "Caballo", "Barco", "Camion"]
        return clases[prediccion]