from typing import Text
import dash
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask, Response
import cv2
import time
import datetime
import numpy as np
import threading

MV_AVR_WIDTH = 3
SQUARE_THRESHOLD = 800


def frame_postprocess(frame, method="MORPH"):
    if method == "MORPH":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    elif method == "BLUR":
        return cv2.blur(frame, (9, 9))


class data_queue(object):
    def __init__(self, SIZE):
        self.currsize = 0
        self.maxsize = SIZE if SIZE > 0 else 0
        data = []
        for i in range(self.maxsize):
            data.append(0)
        self.data = data

    def __str__(self):
        string = ''
        for i in range(self.currsize):
            string += str(self.data[i])
            string += ' '
        return string

    def get_mean(self):
        runningsum = 0
        if self.currsize != 0:
            try:
                runningsum = np.zeros(self.data[0].shape)
            except:
                pass
            for i in range(self.currsize):
                runningsum += self.data[i]
            return runningsum / self.currsize
        else:
            return 0

    def get_data(self):
        return self.data[0:self.currsize]

    def add_data(self, newdata):
        if self.currsize < self.maxsize:
            self.data[self.currsize] = newdata
            self.currsize += 1
        else:
            new_data_list = []
            for i in range(self.maxsize-1):
                new_data_list.append(self.data[i+1])
            new_data_list.append(newdata)
            self.data = new_data_list
        return


class VideoCamera(object):
    def __init__(self, dev_mode=False):
        self.dev_mode = dev_mode
        self.video = cv2.VideoCapture(0)
        self.bkg = cv2.createBackgroundSubtractorMOG2()
        self.frame_queue = data_queue(MV_AVR_WIDTH)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.time = datetime.datetime.now()
        self.detect_counter = 0
        self.bounds = []

    def __del__(self):
        self.video.release()

    def get_frame(self):
        time.sleep(1/(self.fps//2))
        success, frame = self.video.read()
        frame = cv2.resize(frame, (640, 480))
        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detect_counter == self.fps-1:
            bkg_diff = self.bkg.apply(bw_frame)
            self.frame_queue.add_data(bkg_diff)
            curr_frame = self.frame_queue.get_mean()
            curr_frame_eq = cv2.equalizeHist(curr_frame.astype(np.uint8))
            curr_frame_eq = frame_postprocess(curr_frame_eq)
            self.bounds = cv2.findContours(
                curr_frame_eq, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(self.bounds) > 0:
            for j in self.bounds[0]:
                x, y, w, h = cv2.boundingRect(np.array(j))
                if w*h > SQUARE_THRESHOLD:
                    cv2.rectangle(frame, (x, y),
                                  (x+w, y+h), (153, 0, 0), 5)
                    cv2.rectangle(frame, (x, y),
                                  (x+w, y+h), (153, 0, 0), 5)
        ret, jpeg = cv2.imencode('.jpg', frame)
        self.detect_counter = (self.detect_counter + 1) % self.fps
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(
    __name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(dev_mode=True)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
