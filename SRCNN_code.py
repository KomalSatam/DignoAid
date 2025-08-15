import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load preprocessed image
input_path = r'C:\Users\Asus\OneDrive\Documents\SY\Mini project\Processed images\resized_image.png'
img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img)

# Normalize Y channel
y = y.astype('float32') / 255.0
y = np.expand_dims(y, axis=0)      # add batch dimension
y = np.expand_dims(y, axis=-1)     # add channel dimension

# Build SRCNN Model
model = Sequential()
model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)))
model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
model.add(Conv2D(1, (5, 5), activation='linear', padding='same'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Create low-resolution version from Y channel
original_y = y[0, :, :, 0]  # shape (height, width)

# Downscale to 1/2 size (not too aggressive)
h, w = original_y.shape
low_res = cv2.resize(original_y, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

# Upscale back to original size
low_res = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)

# Normalize and reshape for model
low_res_input = np.expand_dims(np.expand_dims(low_res, axis=0), axis=-1)  # (1, h, w, 1)

# Now train model to learn: low_res → original_y
model.fit(low_res_input, y, epochs=100, verbose=1)


# Predict enhanced image
pred = model.predict(y)
pred = np.squeeze(pred)  # remove batch and channel dimensions

# Post-process result
pred = (pred * 255.0).clip(0, 255).astype('uint8')
result = cv2.merge([pred, cr, cb])
final_image = cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)

# Save result
cv2.imwrite(r'C:\Users\Asus\OneDrive\Documents\SY\Mini project\Preprocessed images\enhanced_image.png', final_image)


plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Enhanced")
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
