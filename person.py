import numpy as np
import math
import cv2 as cv

PARTS = {
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


class Person():
    def __init__(self, heatmap, offsets):
        self.keypoints = self.get_keypoints(heatmap, offsets)
        self.pose = self.infer_pose(self.keypoints)

    def get_keypoints(self, heatmaps, offsets, output_stride=32):
        scores = sigmoid(heatmaps)
        num_keypoints = scores.shape[2]
        heatmap_positions = []
        offset_vectors = []
        confidences = []
        for ki in range(0, num_keypoints):
            x, y = np.unravel_index(
                np.argmax(scores[:, :, ki]), scores[:, :, ki].shape)
            confidences.append(scores[x, y, ki])
            offset_vector = (offsets[y, x, ki],
                             offsets[y, x, num_keypoints + ki])
            heatmap_positions.append((x, y))
            offset_vectors.append(offset_vector)
        image_positions = np.add(
            np.array(heatmap_positions) *
            output_stride,
            offset_vectors)
        keypoints = [KeyPoint(i, pos, confidences[i])
                     for i, pos in enumerate(image_positions)]
        return keypoints

    def infer_pose(self, coords):
        return "Unknown"

    def get_coords(self):
        return [kp.point() for kp in self.keypoints]  # if kp.confidence > 0.8

    def get_limbs(self):
        pairs = [
            (5, 6),
            (5, 7),
            (7, 9),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 8),
            (8, 10),
            (6, 12),
            (12, 14),
            (14, 16),
            (11, 12)
        ]
        # if (self.keypoints[i].confidence > 0.8 and
        # self.keypoints[j].confidence > 0.8)
        limbs = [(self.keypoints[i].point(), self.keypoints[j].point())
                 for i, j in pairs]
        return list(filter(lambda x: x is not None, limbs))

    def confidence(self):
        return np.mean([k.confidence for k in self.keypoints])

    def to_string(self):
        return "\n".join([a.to_string() for a in self.keypoints])


class KeyPoint():
    def __init__(self, index, pos, v):
        x, y = pos
        self.x = x
        self.y = y
        self.index = index
        self.body_part = PARTS.get(index)
        self.confidence = v

    def point(self):
        return int(self.y), int(self.x)

    def to_string(self):
        return 'part: {} location: {} confidence: {}'.format(
            self.body_part, (self.x, self.y), self.confidence)


# npz = np.load('sample3.npz')
# data = npz['arr_0']
# offsets = npz['arr_1']
# person = Person(data, offsets)
# print(person.to_string())
