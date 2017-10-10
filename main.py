import argparse
import cv2
import numpy as np
import imageio
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='process Nishika Photos')
parser.add_argument('--input', '-i',
					default='in.jpg',
					help='input file containing Nishika pictures')
parser.add_argument('--output', '-o',
					default='out.gif',
					help='output gif file')
parser.add_argument('--height', '-y',
					default=1400, type=int,
					help='scaled height of the output image.')
parser.add_argument('--num_frames', '-n',
					default=4, type=int,
					help='Number of frames in the Nishika picture.')
parser.add_argument('--boomerang', '-z',
					default=True, type=bool,
					help='Whether the gif should boomerang back-and-forth.')

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

    def findCrop(self):
        """
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
        _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.boundingRect(cnt)
        """
        hborder, vborder = 0.05, 0.01
        rect = (int(self.width * hborder), int(self.height * vborder),
                int(self.width * (1 - 2 * hborder)), int(self.height * (1 - 2 * vborder)))
        return rect

    def crop(self, rect):
        x,y,w,h = rect
        return Frame(self.img[y:y+h,x:x+w])


def CommonCrop(rects):
    x,y,w,h = rects[0]
    x2, y2 = x + w, y + h
    for rect in rects:
        x = max(x, rect[0])
        y = max(x, rect[1])
        x2 = min(x2, rect[0] + rect[2])
        x2 = min(x2, rect[1] + rect[3])
    return x, y, x2-x, y2-y


def POI_frame_index(width, num_images):
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
                                 flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
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

    bounding_rects = [a.findCrop() for a in aligned]
    crop_rect = CommonCrop(bounding_rects)
    print 'crop', crop_rect

    #return aligned
    return [a.crop(crop_rect) for a in aligned]


def outputToGif(filename, frames, boomerang=True):
    if boomerang:
        rev = frames[1:-1]
        rev.reverse()
        frames = frames[:] + rev
    # weird index converts from BGR to RGB
    imageio.mimsave(filename, [f.img[...,::-1] for f in frames],
                    duration=0.1)

                    
def main():
  args = parser.parse_args()
  src = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
  h, w = src.shape[0:2]
  target_height = args.height or h
  scale = float(args.height) / h
  scaled = cv2.resize(src, (int(w * scale), target_height),
                      None, interpolation=cv2.INTER_CUBIC)
  frames = sliceAndAlignImages(scaled, args.num_frames)
  outputToGif(args.output, frames, boomerang=args.boomerang)
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
