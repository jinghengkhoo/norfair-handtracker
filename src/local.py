import cv2
from norfair import Tracker
from norfair.distances import create_keypoints_voting_distance

from hands import run
from utils import detector_utils as detector_utils

detection_graph, sess = detector_utils.load_inference_graph()

trackers = {}

if __name__ == "__main__":

    vid = cv2.VideoCapture(0)

    ret, image_np = vid.read()
    keypoint_distance_threshold = image_np.shape[0] / 40
    tracker = Tracker(
                        distance_function=create_keypoints_voting_distance(
                            keypoint_distance_threshold=keypoint_distance_threshold,
                            detection_threshold=0.3,
                        ),
                        distance_threshold=0.4,
                        detection_threshold=0.3,
                        initialization_delay=4,
                        hit_counter_max=30,
                        pointwise_hit_counter_max=10,
                    )

    while(True):
        ret, image_np = vid.read()

        res = run(detection_graph, sess, trackers["id"], image_np)

        print(res)