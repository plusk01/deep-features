import matplotlib.pyplot as plt
import scipy.misc as scim
import numpy as np
import cv2

cap = cv2.VideoCapture('campus.mp4')

frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print("Recorded fps: {}".format(fps))

save_at_frames = [0, 100, 200, 300]

for ii in xrange(frame_count):
    ret, frame = cap.read()
    
    if ii in save_at_frames:

        # scale frame by a percentage
        img = scim.imresize(frame, 0.5)

        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Let the user know what's going on
        filename = 'frames/frame_%d.png'%ii
        print("Saving {}...".format(filename))

        # Save to file
        scim.imsave(filename, img)

    if ii > max(save_at_frames):
        break

cv2.destroyAllWindows()
cap.release()