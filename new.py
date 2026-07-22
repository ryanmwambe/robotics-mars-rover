from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

import cv2
import time

from imx500_object_detection_demo import (
    parse_detections,
    get_labels
)


app = Flask(__name__)


MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


print("Loading IMX500 model")

imx500 = IMX500(MODEL)

intrinsics = imx500.network_intrinsics


if intrinsics is None:
    print("No network intrinsics found")
    exit()


print("Loading labels")

labels = get_labels()



picam2 = Picamera2()


config = picam2.create_preview_configuration(
    main={
        "size": (640,480),
        "format": "RGB888"
    }
)


config["post_processing"] = {
    "post_processing_file":
    "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
}



picam2.configure(config)


print("Starting camera")

picam2.start()


time.sleep(3)

print("Camera started")



def generate():


    while True:


        frame = picam2.capture_array()


        metadata = picam2.capture_metadata()



        detections = parse_detections(
            metadata,
            picam2,
            labels,
            intrinsics
        )



        for detection in detections:


            x,y,w,h = detection.box



            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )



            text = (
                f"{detection.name} "
                f"{detection.confidence:.2f}"
            )


            cv2.putText(
                frame,
                text,
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )



        frame=cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


        success,jpeg=cv2.imencode(
            ".jpg",
            frame
        )


        if success:


            yield(
                b"--frame\r\n"
                b"Content-Type:image/jpeg\r\n\r\n"
                +
                jpeg.tobytes()
                +
                b"\r\n"
            )




@app.route("/")
def home():

    return """
    <h1>IMX500 Detection</h1>
    <img src="/video">
    """



@app.route("/video")
def video():

    return Response(
        generate(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )



if __name__=="__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
