from sys import flags
import cv2
import time
import datetime
import numpy as np
import dlib
import cam_settings

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_points = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               26, 25, 24, 23, 22, 21, 20, 19,
               18, 17]


def get_face_region(facial_landmarks, face_points=face_points):
    def g(i): return (facial_landmarks.part(
        face_points[i]).x, facial_landmarks.part(face_points[i]).y)
    face_region = np.array(list(map(g, np.arange(0, 27))))
    x_, y_ = np.min(face_region[:, 0]), np.min(face_region[:, 1])
    x1_, y1_ = np.max(face_region[:, 0]), np.max(face_region[:, 1])
    return [(x_, y_), (x1_, y1_)]


def disable_background(img, facial_landmarks, face_points=face_points):
    def g(i): return (facial_landmarks.part(
        face_points[i]).x, facial_landmarks.part(face_points[i]).y)
    face_region = np.array(list(map(g, np.arange(0, 27))))
    np.max(face_region[:, 0])
    height_, width_ = img.shape
    mask = np.zeros((height_, width_), np.uint8)
    cv2.polylines(mask, [face_region], True, 255, 2)
    cv2.fillPoly(mask, [face_region], 255)
    img = cv2.bitwise_and(img, img, mask=mask)
    avr = cv2.mean(img, mask=mask)[0]
    avg_back = (255-mask)/255*avr
    img = img+avg_back
    x_, y_ = np.min(face_region[:, 0]), np.min(face_region[:, 1])
    x1_, y1_ = np.max(face_region[:, 0]), np.max(face_region[:, 1])
    return img[max(y_, 0):min(y1_, height_), max(0, x_):min(x1_, width_)]


def frame_preprocess(frame):
    return cv2.blur(frame, (9, 9))


def frame_postprocess(frame, method="MORPH"):
    if method == "MORPH":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    elif method == "BLUR":
        return cv2.blur(frame, (13, 13))


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
        self.frame_queue = data_queue(cam_settings.MV_AVR_WIDTH)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.time = datetime.datetime.now()
        self.detect_counter = 0
        self.bounds = []
        self.face_bounds = []
        self.frame = np.zeros((640, 480))
        self.found_movement = False
        if dev_mode:
            print("FPS: %s \nQueue: %s \nStarts at %s" %
                  (self.fps, cam_settings.MV_AVR_WIDTH, self.time))
        self.update_frame()

    def __del__(self):
        self.video.release()

    def __str__(self):
        self.time = datetime.datetime.now()
        self.brightness_detection()
        return "%s\nMovement = %s, Faces = %s, Brightness = %s" % (
            self.time.__str__().split(".")[0], self.found_movement, len(self.face_bounds), self.avr_brightness)

    def update_frame(self):
        success, frame = self.video.read()
        frame = cv2.resize(frame, (640, 480))
        if self.detect_counter == (self.fps-1):
            bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cam_settings.DO_PRE_PROCESS:
                bw_frame = frame_preprocess(bw_frame)
            bkg_diff = self.bkg.apply(bw_frame)
            self.frame_queue.add_data(bkg_diff)
            curr_frame = self.frame_queue.get_mean()
            curr_frame_eq = cv2.equalizeHist(curr_frame.astype(np.uint8))
            curr_frame_eq = frame_postprocess(curr_frame_eq, method="BLUR")
            self.bounds = cv2.findContours(
                curr_frame_eq, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            s = 0
            for j in self.bounds[0]:
                x, y, w, h = cv2.boundingRect(np.array(j))
                s += w*h
            if s > cam_settings.MOVE_THRESHOLD:
                self.found_movement = True
                self.get_face()
        if len(self.bounds) > 0:
            for j in self.bounds[0]:
                x, y, w, h = cv2.boundingRect(np.array(j))
                if w*h > cam_settings.SQUARE_THRESHOLD:
                    cv2.rectangle(frame, (x, y),
                                  (x+w, y+h), (153, 0, 0), 5)
        if len(self.face_bounds) > 0:
            for bound in self.face_bounds:
                cv2.rectangle(frame, bound[0],
                              bound[1], (0, 153, 0), 5)
        self.frame = frame
        self.detect_counter = (self.detect_counter + 1) % self.fps
        time.sleep(1/(self.fps//2))
        return

    def get_frame(self):
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def get_frame_orig(self):
        return self.frame

    def get_flag(self):
        return self.found_movement

    def get_face(self):
        #frame_faces_eq = cv2.equalizeHist(self.frame)
        frame_faces = detector(self.frame)
        if frame_faces.__len__() != 0:
            face_bounds = []
            for face in frame_faces:
                landmarks = predictor(self.frame, face)
                face_loc = get_face_region(landmarks)
                face_bounds.append(face_loc)
            self.face_bounds = face_bounds
        else:
            self.face_bounds = []
        return

    def brightness_detection(self):
        bw_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        bw_shape = bw_frame.shape
        self.avr_brightness = int(np.sum(bw_frame)/(bw_shape[0]*bw_shape[1]))
        return self.avr_brightness
