import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Hook into OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        # Normalize the depth map for display
        output = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    # Convert depth map to 8-bit grayscale for display
    output_gray = (255 * output.cpu().numpy()).astype(np.uint8)

    # Apply a colormap for better visualization
    output_colored = cv2.applyColorMap(output_gray, cv2.COLORMAP_PLASMA)

    # Display the depth map
    cv2.imshow('Depth Map', output_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
