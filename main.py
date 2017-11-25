import argparse
import cv2
import numpy as np
import imageio
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='process Nishika Photos')
parser.add_argument('--input', '-i',
          nargs='+',
					help='input file containing Nishika pictures')
parser.add_argument('--pattern', '-p',
					help='input files of pattern <myfilename>_i.jpg, from i between 1 and n')
parser.add_argument('--output', '-o',
					default='out.gif',
					help='output gif file')
parser.add_argument('--height', '-y',
					default=1400, type=int,
					help='scaled height of the output image.')
parser.add_argument('--frames', '-n',
					default=0, type=int,
					help='Number of frames in the Nishika picture. If len(input) == 1, and frames > 1, then the input image will be split.')
parser.add_argument('--boomerang', '-z',
					default=True, type=bool,
					help='Whether the gif should boomerang back-and-forth.')

POI_frame_index = 0
POI_frame_offset = [0,0]

class Frame:
  def __init__(self, img):
    self.img = img
    self._gray = None
    self._scaled = None
    self.height, self.width = img.shape[:2]
    self.scale = 400.0 / self.height
    self.scaled_h, self.scaled_w, = int(self.height * self.scale), int(self.width * self.scale)
      
  def getScaled(self):
    if self._scaled is None:
      self._scaled = cv2.resize(self.img, (int(self.width * self.scale), 400),
                                None, interpolation=cv2.INTER_CUBIC)
    return self._scaled

  def getGray(self):
    if self._gray is None:
      self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    return self._gray

  def findCrop(self):
    hborder, vborder = 0.05, 0.01
    rect = (int(self.width * hborder), int(self.height * vborder),
            int(self.width * (1 - 2 * hborder)), int(self.height * (1 - 2 * vborder)))
    return rect

  def crop(self, rect):
    x,y,w,h = rect
    return Frame(self.img[y:y+h,x:x+w])
      
  def POI_frame_index(self, num_frames):
    return int((POI[0] / float(self.width)) * num_frames)
  
  def POI_frame_offset(self, num_frames, edge_padding=32):
    clamp = lambda low, x, high: max(low, min(x, high))
    frame_width = int(self.width / num_frames)
    x = clamp(edge_padding,
              POI[0] - (self.POI_frame_index(num_frames) * frame_width),
              frame_width - edge_padding)
    y = clamp(edge_padding,
              POI[1],
              self.height - edge_padding)
    return int(x), int(y)
    
  def POI_location(self, num_frames, edge_padding=32):
    f_index = self.POI_frame_index(num_frames)
    off = self.POI_frame_offset(num_frames, edge_padding)
    return ((int(self.width / (num_frames)) * f_index) + off[0], off[1])


def CommonCrop(rects):
  x,y,w,h = rects[0]
  x2, y2 = x + w, y + h
  for rect in rects:
    x = max(x, rect[0])
    y = max(x, rect[1])
    x2 = min(x2, rect[0] + rect[2])
    x2 = min(x2, rect[1] + rect[3])
  return x, y, x2-x, y2-y

            
def sliceFrames(src_frame, num_images):
  w, h = src_frame.width, src_frame.height
  frames = [Frame(src_frame.img[0:h, i*w/4:(i+1)*w/4])
            for i in range(0, 4)]
  return frames

    
def alignFrames(frames, poi_frame, poi_offset, poi_stride, warp_mode=cv2.MOTION_TRANSLATION):
  ret = []
  poi_top_left = int(poi_offset[0] - poi_stride/2), int(poi_offset[1] - poi_stride/2)
  poi_img = frames[poi_frame].img[poi_top_left[1]:poi_top_left[1] + poi_stride,
                                  poi_top_left[0]:poi_top_left[0] + poi_stride]
  templ_gray = cv2.cvtColor(poi_img, cv2.COLOR_BGR2GRAY)
  
  for i in range(len(frames)):
    if i == poi_frame:
      ret.append(frames[i])
      continue
    ref = frames[i]
    res = cv2.matchTemplate(ref.getGray(), templ_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    frame_offset = (poi_top_left[0] - max_loc[0], poi_top_left[1] - max_loc[1])
    M = np.float32([[1,0,frame_offset[0]],
                    [0,1,frame_offset[1]]])
    aligned = cv2.warpAffine(frames[i].img, M, (ref.width, ref.height))
    ret.append(Frame(aligned))

  return ret

    
def getPOIInput(frames):
  cv2.namedWindow('image')
  
  POI_frame_index = 0
  POI_location = [0, 0]
  bound = {'POI': [0,0]}
  #scaled_frames = frames
  framebuffer = np.zeros((max([f.height for f in frames]),
                          sum([f.width for f in frames]),3), np.uint8)

  def _capture_focus(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
      for frame in frames:
        pass
      bound['POI'] = [x, y]
  
  cv2.setMouseCallback('image', _capture_focus)

  POI_STRIDE = 64
  while True:

    #img = Frame(src_frame.img.copy())

    x_offset = 0
    for i in range(0, len(frames)):
      # draw images to framebuffer
      framebuffer[0:frames[i].height, x_offset:x_offset+frames[i].width] = frames[i].img
      x_offset += frames[i].width

      # draw sample slice lines
      cv2.line(framebuffer, (x_offset, 0),
               (x_offset, frames[i].height),
               (0, 0, 255))

    # draw POI
    POI = bound['POI']
    if POI:
      cv2.circle(framebuffer, (POI[0], POI[1]), radius=10, color=(0, 0, 255))
      cv2.rectangle(framebuffer, 
                    (POI[0] - int(POI_STRIDE/2), POI[1] - int(POI_STRIDE/2)),
                    (POI[0] + int(POI_STRIDE/2), POI[1] + int(POI_STRIDE/2)), color = (0, 0, 255),
                    thickness=4)
        

    cv2.imshow('image', framebuffer)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
      break
      

  # find frame and offset in frame of Point Of Interest
  POI_acc = 0
  for i in range(0, len(frames)):
    if POI_acc + frames[i].width >= POI[0]:
      print POI[0]
      POI_frame_index = i
      POI_location = [POI[0] - POI_acc, POI[1]]
      break
    POI_acc += frames[i].width
    
  return POI_frame_index, POI_location


def alignImages(frames, poi_frame_index, poi_offset):
    aligned = alignFrames(frames, poi_frame_index, poi_offset, poi_stride=64)

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
  global SCALE
  args = parser.parse_args()
  inputs = args.input
  
  if args.pattern:
    inputs = [args.pattern + '_' + str(i) + '.jpg' for i in range(1, args.frames + 1)]
  
  # split an image if only one passed in
  if len(inputs) == 1 and args.frames > 0:
    src = cv2.imread(inputs[0], cv2.IMREAD_UNCHANGED)
    frames = sliceFrames(Frame(src), args.frames)
  else:
    frames = [Frame(cv2.imread(f, cv2.IMREAD_UNCHANGED)) for f in inputs]
  
  h, w = frames[0].height, frames[0].width
  target_height = args.height or h
  scale = float(target_height) / h
  scaled = [Frame(cv2.resize(f.img, (int(w * scale), target_height),
                             None, interpolation=cv2.INTER_CUBIC)) for f in frames]
  index, offset = getPOIInput(scaled)
  oframes = alignImages(scaled, index, offset)
  outputToGif(args.output, oframes, boomerang=args.boomerang)
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
