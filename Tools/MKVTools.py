import json
import os.path
import time

import pyk4a

from Tools.mkv_reader import MKVReader, TRACK
from tkinter import Tk
from tkinter.filedialog import askopenfilenames, askdirectory
import cv2

from pyk4a import PyK4APlayback

class MKVFile:
    def __init__(self, filename, track_filter):
        reader = MKVReader(filename, track_filter = track_filter)
        self.frames = list()
        self.filename = reader.filename.split('.')[0]
        self.calib = reader.get_calibration()
        while True:
            try:
                f = reader.get_next_frameset()
            except EOFError:
                break

            try:
                color = f[TRACK.COLOR]
                ir = f[TRACK.IR]
                depth = f[TRACK.DEPTH]
            except KeyError:
                print(f"Incomplete Frameset #{f['index'] + 1}! - Skipping to Next Frameset")
                continue


            self.frames.append(Frame(color, ir, depth))

    def save(self, location):
        print(location)
        try:
            os.mkdir(f"{location}/{self.filename}_RAW")
            print(f"Saving to {location}/{self.filename}_RAW - New Directory")
        except FileExistsError:
            print(f"Saving to {location}/{self.filename}_RAW - Existing Directory")

        for i, f in enumerate(self.frames):
            path = f"{location}/{self.filename}_RAW/frame{i}.json"
            print(path)
            with open(path, "w") as outfile:
                outfile.write(f.convert2json())

    def getFrame(self, frame):
        return self.frames[frame]

    def getFrameJson(self, frame):
        return self.getFrame(frame).convert2json()

    def show(self):
        for f in self.frames:
            f.show()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                 break

            time.sleep(1/30)


class Frame:
    def __init__(self, rgb, ir, depth):
        self.rgb = rgb
        self.ir = ir
        self.depth = depth

    def show(self):
        cv2.imshow("RGB_image", self.rgb)
        cv2.imshow("IR_image", self.ir)
        cv2.imshow("Depth_image", self.depth)

    def getRGB(self):
        return self.rgb

    def getIR(self):
        return self.ir

    def getDEPTH(self):
        return self.depth

    def convert2json(self):
        d = dict()
        d['rgb'] = self.rgb.tolist() if self.rgb != [] else self.rgb
        d['ir'] = self.ir.tolist() if self.ir != [] else self.ir
        d['depth'] = self.depth.tolist() if self.depth != [] else self.depth
        return json.dumps(d)


class MKVTools:
    def __init__(self, track_filter=[TRACK.COLOR, TRACK.DEPTH, TRACK.IR]):
        self.track_filter = track_filter
        Tk().withdraw()
        self.filenames = askopenfilenames(defaultextension = '.mkv')
        for f in self.filenames:
            print(f)

    def readfile(self, filename=None):
        if filename == None:
            filename = self.filenames[0]
        return MKVFile(filename, self.track_filter)

    def save(self):
        location = askdirectory()
        for f in self.filenames:
            self.readfile(filename = f).save(location)

    def saveFrame(self, filename, frame):
        location = askdirectory()
        json = self.readfile(filename=filename).getFrameJson(frame)

    def getFrameJson(self, filename, frame):
        return self.readfile(filename=filename).getFrameJson(frame)


if __name__ == '__main__':
    print()



