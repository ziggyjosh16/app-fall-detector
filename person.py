import numpy as np
import math
import cv2 as cv

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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def get_keypoints2(heatmaps, offsets, output_stride=32):
    scores = sigmoid(heatmaps)
    print('scores shape', scores.shape)
    num_keypoints = scores.shape[2]
    heatmap_positions = []
    offset_vectors = []
    for ki in range(0, num_keypoints ):
        x,y = np.unravel_index(np.argmax(scores[:,:,ki]), scores[:,:,ki].shape)
        offset_vector = (offsets[y,x,ki], offsets[y,x,num_keypoints+ki])
        heatmap_positions.append((x,y))
        offset_vectors.append(offset_vector)
        # print("keypoint index: {}, position:{}".format(ki, position))
    image_positions = np.add(np.array(heatmap_positions) * output_stride, offset_vectors)
    print(image_positions)
class Person():
    def __init__(self, heatmap):
        self.keypoints = self.get_keypoints(heatmap)
        self.pose = self.infer_pose(self.keypoints)
    def get_keypoints(self, data):
        height, width, num_keypoints = data.shape
        keypoints = []
        for keypoint in range(0, num_keypoints):
            maxval = data[0][0][keypoint]
            maxrow = 0
            maxcol = 0
            for row in range(0, width):
                for col in range(0,height):
                    if data[row][col][keypoint] > maxval:
                        maxrow = row
                        maxcol = col
                        maxval = data[row][col][keypoint]
            keypoints.append(KeyPoint(keypoint, maxrow, maxcol, maxval))
            # keypoints = [Keypoint(x,y,z) for x,y,z in ]
        return keypoints
    def get_image_coordinates_from_keypoints(self, offsets):
        height, width, depth = (257,257,3)
        # [(x,y,confidence)]
        coords = [{ 'point': k.body_part,
                    'location': (k.x / (width - 1)*width + offsets[k.y][k.x][k.index],
                   k.y / (height - 1)*height + offsets[k.y][k.x][k.index]),
                    'confidence': k.confidence}
                 for k in self.keypoints]
        return coords
    def infer_pose(self, coords):
        return "Unknown"
    def to_string(self):
        return [a.to_string() for a in self.keypoints]


class KeyPoint():
    def __init__(self, index, x, y, v):
        self.x = x
        self.y = y
        self.index = index
        self.body_part = parts.get(index)
        self.confidence = sigmoid(v)
    def to_string(self):
        return '{}\nlocation:{}\nconfidence:{}\n'.format(self.body_part, (self.x, self.y), self.confidence)



npz = np.load('sample3.npz')
data = npz['arr_0']
offsets = npz['arr_1']
# print(offsets)
# person = Person(data)
# [print(a) for a in person.to_string()]
# cs = person.get_image_coordinates_from_keypoints(offsets)
# [print(c) for c in cs]
print(data.shape, offsets.shape)
get_keypoints2(data, offsets)
