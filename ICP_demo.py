from typing import Text
import dash
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask, Response
from threading import Thread
from cam_inst import VideoCamera
import time

vid = VideoCamera(dev_mode=True)


def init_monitor():
    global vid
    while(True):
        vid.update_frame()


def init_server():
    app.run_server(debug=False, host='0.0.0.0', port=8050)


def gen():
    global vid
    while True:
        time.sleep(0.1)
        frame = vid.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(
    __name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    t1 = Thread(target=init_monitor)
    t1.start()
    t2 = Thread(target=init_server)
    t2.start()
    t1.join()
    t2.join()
