import sys
import cv2

import norfair
from norfair import Detection, Tracker
from norfair.distances import create_keypoints_voting_distance

# Import openpose
openpose_install_path = (
    "/openpose"  # Insert the path to your openpose instalation folder here
)
try:
    sys.path.append(openpose_install_path + "/build/python")
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
    )
    raise e

# Wrapper implementation for OpenPose detector
class OpenposeDetector:
    def __init__(self, num_gpu_start=None):
        # Set OpenPose flags
        config = {}
        config["model_folder"] = openpose_install_path + "/models/"
        config["model_pose"] = "BODY_25"
        config["logging_level"] = 3
        config["output_resolution"] = "-1x-1"
        config["net_resolution"] = "-1x768"
        config["num_gpu"] = 1
        config["alpha_pose"] = 0.6
        config["render_threshold"] = 0.05
        config["scale_number"] = 1
        config["scale_gap"] = 0.3
        config["disable_blending"] = False

        # If GPU version is built, and multiple GPUs are available,
        # you can change the ID using the num_gpu_start parameter
        if num_gpu_start is not None:
            config["num_gpu_start"] = num_gpu_start

        # Starting OpenPose
        self.detector = op.WrapperPython()
        self.detector.configure(config)
        self.detector.start()

    def __call__(self, image):
        return self.detector.emplaceAndPop(image)

def run(detector, tracker, path):

    imageToProcess = cv2.imread(path)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    detector(op.VectorDatum([datum]))
    detected_poses = datum.poseKeypoints

    if detected_poses is None:
        tracked_objects = tracker.update()

    detections = (
        []
        if not detected_poses.any()
        else [
            Detection(p, scores=s)
            for (p, s) in zip(
                detected_poses[:, :, :2], detected_poses[:, :, 2]
            )
        ]
    )

    tracked_objects = tracker.update(detections=detections)
    norfair.draw_points(imageToProcess, detections)

    norfair.draw_tracked_objects(imageToProcess, tracked_objects)

    res = []
    if tracked_objects:
        for person in tracked_objects:
            # print(person.live_points)
            # print(person.estimate)
            # print(person.id)
            # print(person.label)
            res.append({
                "left_hand": person.estimate[4],
                "right_hand": person.estimate[7],
                "id": person.id
            })

    return res

if __name__ == "__main__":

    # Process Videos
    detector = OpenposeDetector(None)

    # Define constants
    DETECTION_THRESHOLD = 0.1
    DISTANCE_THRESHOLD = 0.4
    INITIALIZATION_DELAY = 4
    HIT_COUNTER_MAX = 30
    POINTWISE_HIT_COUNTER_MAX = 10

    input_path = "/demo/src/COCO_val2014_000000000241.jpg"

    imageToProcess = cv2.imread(input_path)
    KEYPOINT_DIST_THRESHOLD = imageToProcess.shape[0] / 40

    tracker = Tracker(
        distance_function=create_keypoints_voting_distance(
            keypoint_distance_threshold=KEYPOINT_DIST_THRESHOLD,
            detection_threshold=DETECTION_THRESHOLD,
        ),
        distance_threshold=DISTANCE_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
        initialization_delay=INITIALIZATION_DELAY,
        hit_counter_max=HIT_COUNTER_MAX,
        pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
    )

    for i in range(10):
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        detector(op.VectorDatum([datum]))
        detected_poses = datum.poseKeypoints

        if detected_poses is None:
            tracked_objects = tracker.update()

        detections = (
            []
            if not detected_poses.any()
            else [
                Detection(p, scores=s)
                for (p, s) in zip(
                    detected_poses[:, :, :2], detected_poses[:, :, 2]
                )
            ]
        )

        tracked_objects = tracker.update(detections=detections)
        norfair.draw_points(imageToProcess, detections)

        norfair.draw_tracked_objects(imageToProcess, tracked_objects)

        if tracked_objects:
            for person in tracked_objects:
                print(person.live_points)
                print(person.estimate)
                print(person.id)
                print(person.label)
            sys.exit()

        cv2.imwrite("2.png", imageToProcess)