import matplotlib.pyplot as plt
import scipy.misc as scim
import tensorflow as tf
import numpy as np
import cv2

import vgg16

class FindMeasurements(object):
    """docstring for FindFeatures"""
    def __init__(self, deep=False):
        super(FindMeasurements, self).__init__()
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners=100,
                               qualityLevel=0.3,
                               minDistance=7,
                               blockSize=7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize=(15,15),
                          maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # frame number
        self.frame_num = 0

        # Features from last frame
        self.features_d1 = None

        # Should I use Deep Features?
        if deep:
            self.deep = DeepFeatures(save_im=True)
        else:
            self.deep = None


    def find(self, frame):
        # Convert to grey
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_num > 0:
            good_new, good_old = self.lk_optical_flow(grey)
        else:
            good_new = None
            good_old = None

        # Use the current frame to find features for next time
        self.features_d1 = self.detect_features(grey, color_frame=frame)

        # Copy the current frame to be used in optical flow next time
        self.frame_d1 = grey.copy()

        # Increase frame counter
        self.frame_num += 1

        return good_new, good_old


    def detect_features(self, frame, color_frame=None):
        if self.deep:
            return self.deep.detect_features(color_frame)

        # Regular ol' GFTT
        return cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)


    def lk_optical_flow(self, frame):
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.frame_d1, frame, self.features_d1, None, **self.lk_params)

        # Bail if something weird happens
        if p1 is None or st is None or err is None:
            return None, None

        # Return matched current and old points
        return p1[st==1], self.features_d1[st==1]


class DeepFeatures(object):
    """docstring for DeepFeatures"""
    def __init__(self, save_im=False):
        super(DeepFeatures, self).__init__()
        
        # Start tensorflow session
        self.sess = tf.Session()

        # Create a tmp img just to load weights
        opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='opt_img' )
        tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
        
        # Build the VGG computation graph and load in pre-trained weights
        self.vgg = vgg16.vgg16(tmp_img, 'vgg16_weights.npz', self.sess)

        # Which VGG layer to steal activations from?
        self.feature_layer = self.vgg.conv1_2

        # Which filter output of the feature_layer to use?
        self.feature_map = 45

        # Name of this layer
        self.layer_name = self.feature_layer.name.split(':')[0]

        # frame number
        self.frame_num = 0

        # negative padding when extracting fmap
        self.fmap_padding = 5

        self.imsave = save_im

    def get_activations(self, frame):

        ch = 1 if len(frame.shape) == 2 else frame.shape[2]

        # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        r = 224.0 / frame.shape[1]
        dim = (224, int(frame.shape[0] * r)) # flip for cv2.resize
         
        # perform the actual resizing of the image and show it
        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Add black borders and make square (16:9)
        zeros = np.zeros( (224,224,ch) )
        s = (224-img.shape[0])/2
        e = s + img.shape[0]
        zeros[s:e, :, :] = img

        # VGG expects RBG, so if grayscale lets just ... copy the channels?
        # Maybe that's stupid
        if ch == 1:
            zeros[:, :, 1] = zeros[:, :, 0]
            zeros[:, :, 2] = zeros[:, :, 0]


        # Reshape for VGG
        img = np.reshape(zeros, [1,224,224,3] )

        # cv2.imshow("resized", resized)

        # Get the features
        acts = self.sess.run(self.feature_layer, feed_dict={self.vgg.imgs: img})

        return acts, s, e

    def detect_features(self, frame):

        # Get the VGG16 feature map, along with ROI start and end (height)
        acts, roi_sh, roi_eh = self.get_activations(frame)

        if self.imsave and self.frame_num == 100:
            img = self.display_acts(acts)
            scim.imsave('fmaps/acts_%s.png'%self.layer_name, img)

        # increment frame counter
        self.frame_num += 1

        # Normalize and choose the right feature map; selecting only the ROI
        fmap = np.squeeze(acts)
        roi_sh += self.fmap_padding
        roi_eh -= self.fmap_padding
        roi_sw = self.fmap_padding
        roi_ew = frame.shape[1] - self.fmap_padding
        fmap = fmap[roi_sh:roi_eh, roi_sw:roi_ew, self.feature_map]
        fmap = fmap/np.max(fmap)

        # Erode while the image is small
        fmap = cv2.erode(fmap, np.ones( (1,1), dtype=np.uint8), iterations=1)
        fmap = cv2.dilate(fmap, np.ones( (3,3), dtype=np.uint8), iterations=2)

        # make big again so mapping will work
        fmap = cv2.resize(fmap, (frame.shape[1],frame.shape[0]))

        # Turn feature map into acts
        threshold_l = 0.50
        threshold_u = 1.00
        fmap_th = np.logical_and(fmap>threshold_l, fmap<threshold_u)
        idx = np.where(fmap_th)
        yy = idx[0]
        xx = idx[1]

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        Z = np.vstack((xx,yy)).T.astype(np.float32)
        ret, label, center = cv2.kmeans(Z, K, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        

        if self.frame_num % 50 == 0 and False:
            print(len(xx))
            plt.scatter(xx,yy, marker='.'); plt.hold(True)
            plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
            plt.show()

            # import ipdb; ipdb.set_trace()

        # Show the feature map and the thresholded image
        cv2.imshow('fmap',fmap)
        cv2.imshow('fmap_th',fmap_th.astype(np.uint8)*255)

        # Shape features like the output of GFTT
        #features = np.reshape(np.array([xx,yy]).T, [len(xx), 1, 2]).astype(np.float32)

        features = np.reshape(center, [center.shape[0], 1, 2]).astype(np.float32)

        return features[:,:,:]

    def display_acts(self, acts):
        acts = np.squeeze(acts)

        # get size information of activations
        h, w, depth = acts.shape

        # Create a square
        s = int(np.ceil(np.sqrt(depth)))

        # Create a blank image to tile acts onto
        img = np.zeros( (h*s, w*s) )

        for i in xrange(depth):

            # from 1D index create (r,c) coords
            r = i / (s)
            c = i % (s)

            # start and end of height dim
            sh = r*h
            eh = sh + h
            # start and end of width dim
            sw = c*w
            ew = sw + w

            # Tile the image
            img[sh:eh, sw:ew] = acts[:, :, i]/np.max(acts[:, :, i])

            # Add text to indicate which image this is
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '%d'%(i+1), (sw,eh-5), font, 1, 1, 2)

        return img