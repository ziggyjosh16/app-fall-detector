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

class Person():
    def __init__(self, heatmap):
        self.keypoints = self.get_keypoints(heatmap)
        self.pose = self.infer_pose(self.keypoints)
    def get_keypoints(self, data):
        print(data.shape)
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
    def infer_pose(self, keypoints):
        return "Unknown"
    def to_string(self):
        return [a.to_string() for a in self.keypoints]


class KeyPoint():
    def __init__(self, index, x, y, v):
        self.x = x
        self.y = y
        self.body_part = parts.get(index)
        self.confidence = v
    def to_string(self):
        return '{}\nlocation:{}\nconfidence:{}\n'.format(self.body_part, (self.x, self.y), self.confidence)




data = np.load('sample3.npy')

person = Person(data)
[print(a) for a in person.to_string()]