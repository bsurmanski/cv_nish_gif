import cv2
from matplotlib import pyplot as plt

POI = []

class Frame:
	def __init__(self, img):
		self.img = img
		self.height, self.width = img.shape[:2]
		self.scale = 400.0 / self.height
		self.scaled = cv2.resize(self.img, (int(self.width * self.scale), 400), 
								 cv2.INTER_CUBIC)
		self.scaled_h, self.scaled_w, = self.scaled.shape[:2]
		
	def compute(self, algo):
		self.kp, self.des = algo.detectAndCompute(self.scaled, None)

def capture_focus(event, x, y, flags, param):
	global POI
	if event == cv2.EVENT_LBUTTONUP:
		POI = [x, y]
		
def sliceAndAlignImages(src, num_images):
	src_frame = Frame(src)

	cv2.namedWindow('image')
	cv2.setMouseCallback('image', capture_focus)
	
	while True:
		img = src_frame.scaled.copy()
		
		# draw sample slice lines
		for i in range(1, num_images):
			cv2.line(img, 
					 (i*src_frame.scaled_w/4, 0), 
					 (i*src_frame.scaled_w/4, src_frame.scaled_h), 
					 (0, 0, 255))

		# draw POI
		if POI:
			cv2.circle(img, (POI[0], POI[1]), 
					   radius=3, color=(0, 0, 255))

		cv2.imshow('image', img)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			break
	
	w, h = src_frame.width, src_frame.height
	frames = [Frame(src_frame.img[0:h, i*w/4:(i+1)*w/4]) 
			  for i in range(0, 4)]
	orb = cv2.ORB_create()
	for frame in frames:
		frame.compute(orb)
	
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(frames[0].des, frames[1].des)
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawMatches(frames[0].scaled, frames[0].kp, 
						   frames[1].scaled, frames[1].kp, matches, None, flags=2)
	#img2 = cv2.drawKeypoints(frame.img, frame.kp, None, color=(0,0,255), flags=0)
	plt.imshow(img3),plt.show()
	
			
def outputToGif(filename, frames):
	pass

def main():
	src = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)
	
	frames = sliceAndAlignImages(src, 4)
	outputToGif('out.gif', frames)
	
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()