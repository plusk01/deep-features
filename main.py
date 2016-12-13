import numpy as np
import cv2

from tqdm import tqdm

from findmeasurements import FindMeasurements

cap = cv2.VideoCapture('campus.mp4')

frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print("Recorded fps: {}".format(fps))

# Create some random colors
color = np.random.randint(0,255,(100,3))

findMeas = FindMeasurements(deep=True)

# Create a mask image for drawing purposes
mask = np.zeros( (height, width, 3), dtype=np.uint8 )

for ii in tqdm(xrange(frame_count)):
    ret,frame = cap.read()
    
    pts_new, pts_old = findMeas.find(frame)

    if pts_new is not None and pts_old is not None:
        # draw the tracks
        for i,(new,old) in enumerate(zip(pts_new,pts_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i%len(color)].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i%len(color)].tolist(),-1)
    
    img = cv2.add(frame, mask)

    # Show the image
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()