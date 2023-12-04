import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()


transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

   
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        
        output = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    
    output_gray = (255 * output.cpu().numpy()).astype(np.uint8)

    
    output_colored = cv2.applyColorMap(output_gray, cv2.COLORMAP_PLASMA)

    
    cv2.imshow('Depth Map', output_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
