import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
from tqdm import tqdm

import vgg16

NUM_EPOCHS = 6000

#####################################
#  Setup VGG with white-noise input #
#####################################

sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

# Build the VGG computation graph and load in pre-trained weights
with tf.name_scope('VGG16'):
    vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'content.png', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#####################################
## Definition of computation graph ##
#####################################

#
# --- construct your cost function here
#
# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)


# Loss function from paper
with tf.name_scope('loss'):
    with tf.name_scope('content-loss'):
        with tf.name_scope('conv4_2-loss'):
            L_content = tf.nn.l2_loss(vgg.conv4_2 - content_acts[8])

    # Style loss
    with tf.name_scope('style-loss'):
        E = []
        for ell in [0, 2, 4, 7, 10]:
            with tf.name_scope('%s-loss'%layers[ell]):
                shape = style_acts[ell].shape
                N = shape[3] # depth of layer; i.e., how many filters?
                M = shape[1] * shape[2] # width x height of feature map

                # layer from VGG
                F = getattr(vgg, layers[ell])
                F = tf.reshape(F, [M, N])

                # Gramian matrix of image to be generated
                G = tf.matmul(F, F, transpose_a=True)

                # Gramian matrix of original image
                a = style_acts[ell].reshape( (M, N) )
                A = np.matmul(a.T, a)

                eta = 1.0/(2 * N**2 * M**2)
                E_ell = eta*tf.nn.l2_loss(G - A)
                E.append( E_ell )

                tf.scalar_summary('style loss %s'%layers[ell], E_ell)

        # make a np.array
        E = np.array(E)

        # Make a list of weights
        weights = np.repeat([1.0/E.size], E.size)

        # Lstyle, a linear combination of Es
        L_style = None
        for i, w in enumerate(weights):
            tmp = w*E[i]

            if L_style is None:
                L_style = tmp
            else:
                L_style += tmp

    # total loss
    with tf.name_scope('total-loss'):
        alpha = 0.1
        beta = 100

        L_total = alpha*L_content + beta*L_style

#####################################
###  Training and Model Accuracy  ###
#####################################

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)

with tf.name_scope('optimizer'):
    # optimizer = tf.train.AdamOptimizer(0.1)
    # train_op = optimizer.minimize(L_total, var_list=[opt_img])



    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    learning_rate = tf.train.exponential_decay(
      0.1,              # Base learning rate.
      batch,            # Current index into the dataset.
      NUM_EPOCHS,       # Decay step.
      0.95,             # Decay rate.
      staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(
                    L_total, global_step=batch, var_list=[opt_img])



# Create summaries to allow us to have introspection into our model
tf.scalar_summary('total loss', L_total)
tf.scalar_summary('style loss', L_style)
tf.scalar_summary('content loss', L_content)

# Merge all the summaries we made for easy access later
merged = tf.merge_all_summaries()

#####################################
### Running the Computation Graph ###
#####################################

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)

# --- place your optimization loop here

# Print table headers like David
print("ITER\t LOSS\t\t\tSTYLE LOSS\t  CONTENT LOSS")

for i in tqdm(xrange(NUM_EPOCHS+1)):

    # Train the twins using the given batch
    summary, _ = sess.run([merged, train_op])#,
                    # feed_dict={
                    #     vgg.imgs: opt_img
                    # })
   
    # Actually add the summary to the tensorboard for visualization
    summary_writer.add_summary(summary, i)

    if i % 100 == 0:
        t_loss, s_loss, c_loss = sess.run([L_total, L_style, L_content])
        # Make a print out to compare with David's
        print("{}\t {}\t\t\t{}\t  {}".format(i, t_loss, s_loss, c_loss))

        # Clip image and put reassign it.
        img = sess.run(opt_img)
        img = tf.clip_by_value( img, 0.0, 255.0 )
        sess.run( opt_img.assign( img ))

# Save the image
img = sess.run(opt_img)
imsave('out.png', img.reshape( (224,224,3) ))