import cv2
import numpy as np
import imageio
from matplotlib import pyplot as plt

POI = []

class Frame:
    def __init__(self, img):
        self.img = img
        self._gray = None
        self.height, self.width = img.shape[:2]
        self.scale = 400.0 / self.height
        self.scaled = cv2.resize(self.img, (int(self.width * self.scale), 400),
                                 None, interpolation=cv2.INTER_CUBIC)
        self.scaled_h, self.scaled_w, = self.scaled.shape[:2]

    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def compute(self, algo):
        self.kp, self.des = algo.detectAndCompute(self.scaled, None)

    def cropBorder(self):
        return self.img
        pass
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        return self.img[y:y+h,x:x+w]


def POI_frame_index(width, num_images):
    print "POI ", POI[0], float(width), num_images
    return int((POI[0] / float(width)) * num_images)


def capture_focus(event, x, y, flags, param):
    global POI
    if event == cv2.EVENT_LBUTTONUP:
            POI = [x, y]

def sliceFrames(src_frame, num_images):
    w, h = src_frame.width, src_frame.height
    frames = [Frame(src_frame.img[0:h, i*w/4:(i+1)*w/4])
              for i in range(0, 4)]
    return frames

def alignFrames(frames, warp_mode=cv2.MOTION_TRANSLATION):
    poi_frame = POI_frame_index(sum([f.scaled_w for f in frames]), len(frames))
    ret = []
    for i in range(len(frames)):
        print("align ", i)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        if i == poi_frame:
            ret.append(frames[i])
            continue
        ref = frames[poi_frame]
        cv2.findTransformECC(ref.gray(), frames[i].gray(), warp_matrix, warp_mode)
        aligned = cv2.warpAffine(frames[i].img, warp_matrix, (ref.width, ref.height),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        ret.append(Frame(aligned))

    return ret


def sliceAndAlignImages(src, num_images):
    global POI

    src_frame = Frame(src)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', capture_focus)

    while True:
        img = src_frame.scaled.copy()

        # draw sample slice lines
        for i in range(1, num_images):
            cv2.line(img, (i*src_frame.scaled_w/4, 0),
                     (i*src_frame.scaled_w/4, src_frame.scaled_h),
                     (0, 0, 255))

        # draw POI
        if POI:
            cv2.circle(img, (POI[0], POI[1]), radius=3, color=(0, 0, 255))

        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    frames = sliceFrames(src_frame, num_images)
    aligned = alignFrames(frames)

    return aligned


def outputToGif(filename, frames, boomerang=True):
    if boomerang:
        rev = frames[1:-1]
        rev.reverse()
        frames = frames[:] + rev
    # weird index converts from BGR to RGB
    imageio.mimsave(filename, [f.cropBorder()[...,::-1] for f in frames],
                    duration=0.1)

def main():
    src = cv2.imread('img_small.jpg', cv2.IMREAD_UNCHANGED)
    frames = sliceAndAlignImages(src, 4)
    outputToGif('out.gif', frames)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
