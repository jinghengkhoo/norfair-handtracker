import logging
import os
import uuid
import shutil
import cv2

from hands import run, OpenposeDetector
from norfair import Tracker
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from norfair.distances import create_keypoints_voting_distance

project_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    CORS_HEADERS='Content-Type'
)

logger = logging.getLogger()

api = Api(app, prefix='/api')

detector = OpenposeDetector(None)

trackers = {}

@app.route("/")
def home():
    return ":D"

class RunInferenceAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        res = {
            "results": {},
            "errors": {},
            "success": False
        }

        data = request.form
        foldername = str(uuid.uuid4()) + ".png"
        folder_path = os.path.join("/demo/src/", foldername)
        upload = request.files["upload"]
        path = os.path.join(folder_path, upload.filename)
        os.makedirs(folder_path)
        upload.save(path)

        if not data["id"] in trackers:
            imageToProcess = cv2.imread(path)
            keypoint_distance_threshold = imageToProcess.shape[0] / 40
            trackers["id"] = Tracker(
                                        distance_function=create_keypoints_voting_distance(
                                            keypoint_distance_threshold=keypoint_distance_threshold,
                                            detection_threshold=0.1,
                                        ),
                                        distance_threshold=0.4,
                                        detection_threshold=0.1,
                                        initialization_delay=4,
                                        hit_counter_max=30,
                                        pointwise_hit_counter_max=10,
                                    )

        res["results"] = run(detector, trackers["id"], path)
        shutil.rmtree(folder_path)

        res["success"] = True
        return res


api.add_resource(RunInferenceAPIView, '/run')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8001, debug=True)