import numpy as np
from norfair import Detection

from utils import detector_utils as detector_utils


def hand_detections_to_norfair_detections(boxes, scores):
    """convert detections_as_xywh to norfair detections"""
    norfair_detections = []


    for idx, detection_as_xyxy in enumerate(boxes):
        bbox_np = np.array(
            [
                [detection_as_xyxy[0], detection_as_xyxy[2]],
                [detection_as_xyxy[1], detection_as_xyxy[3]],
            ]
        )
        scores_np = np.array(
            [scores[idx], scores[idx]]
        )
        norfair_detections.append(
            Detection(
                points=bbox_np, scores=scores_np, label=1
            )
        )

    return norfair_detections

def run(detection_graph, sess, tracker, image_np):

    boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

    detections = hand_detections_to_norfair_detections(boxes, scores)

    if not detections:
        tracked_objects = tracker.update()

    tracked_objects = tracker.update(detections=detections)

    res = []
    if tracked_objects:
        for person in tracked_objects:
            res.append({
                "bbox": person.estimate,
                "id": person.id
            })

    return res
