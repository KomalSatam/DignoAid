import cv2
import numpy as np
image=cv2.imread(r'C:\Users\Asus\OneDrive\Documents\SY\Mini project\Chest_image1.png')
noise_reduced=cv2.medianBlur(image,9)
normalized = noise_reduced / 255.0
resized = cv2.resize(normalized, (128, 128))
resized_uint8=(resized*255).astype('uint8')
cv2.imshow('Resized Normalized Image', (resized * 255).astype('uint8'))
cv2.imshow('Original image:',image)
cv2.imshow('Final image:',normalized)
cv2.imwrite(r'C:\Users\Asus\OneDrive\Documents\SY\Mini project\Processed images\resized_image.png',resized_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()