import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np

parts = {
	0: 'NOSE',
	1: 'LEFT_EYE',
  	2: 'RIGHT_EYE',
  	3: 'LEFT_EAR',
  	4: 'RIGHT_EAR',
  	5: 'LEFT_SHOULDER',
  	6: 'RIGHT_SHOULDER',
  	7: 'LEFT_ELBOW',
  	8: 'RIGHT_ELBOW',
  	9: 'LEFT_WRIST',
  	10: 'RIGHT_WRIST',
  	11: 'LEFT_HIP',
  	12: 'RIGHT_HIP',
  	13: 'LEFT_KNEE',
  	14: 'RIGHT_KNEE',
  	15: 'LEFT_ANKLE',
  	16: 'RIGHT_ANKLE'
}

img = cv.imread('photos\standing\\3.jpg')
print(img.shape)
# img = tf.reshape(img, [1,257,257,3])
img = tf.reshape(tf.image.resize(img, [257,257]), [1,257,257,3])

model = tf.lite.Interpreter('models\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

floating_model = input_details[0]['dtype'] == np.float32

if floating_model:
	img = (np.float32(img) - 127.5) / 127.5


print('input shape: {}, img shape: {}'.format(input_details[0]['shape'], img.shape))
# print(output_details)
# input_data = decode_img(img)

model.set_tensor(input_details[0]['index'], img)
model.invoke()


output_data =  model.get_tensor(output_details[0]['index'])# o()
offset_data = model.get_tensor(output_details[1]['index'])
results = np.squeeze(output_data)
offsets_results = np.squeeze(offset_data)
print("output shape: {}".format(output_data.shape))
np.savez('sample3.npz', results, offsets_results)





# top_k = results.argsort()[-5:][::-1]
# print(top_k)
# for i in top_k:
# 	if floating_model:
# 		print('{:08.6f}: {}'.format(float(results[i]), parts.get(i)))
# 	else:
# 		print('{:08.6f}: {}'.format(float(results[i] / 255.0), parts.get(i)))