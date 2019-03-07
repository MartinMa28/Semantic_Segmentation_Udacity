#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import datetime


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # applies 1*1 convolution on fully-connected layers, 1/32
    conv_1x1_vgg_7 = tf.layers.conv2d_transpose(vgg_layer7_out, filters=num_classes, kernel_size=1,\
        padding = 'same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),\
        name='conv_1x1_vgg_7')
    
    # upsamples deconvolution by 2
    first_upsample_x2 = tf.layers.conv2d_transpose(conv_1x1_vgg_7, filters=num_classes, kernel_size=4,\
        strides=(2, 2), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),\
        name='first_upsample_x2')
    
    # applies 1*1 convolution on vgg 4 output
    conv_1x1_vgg_4 = tf.layers.conv2d_transpose(vgg_layer4_out, filters=num_classes, kernel_size=1,\
        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        name='conv_1x1_vgg_4')
    
    # the first skip layer, 1/16
    first_skip = tf.add(first_upsample_x2, conv_1x1_vgg_4, name='first_skip')

    # upsample deconvolution by 2, again
    second_upsample_x2 = tf.layers.conv2d_transpose(first_skip, filters=num_classes, kernel_size=4,\
        strides=(2, 2), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        name='second_upsample_x2')
    
    # applies 1*1 convolution on vgg3 output
    conv_1x1_vgg_3 = tf.layers.conv2d_transpose(vgg_layer3_out, filters=num_classes, kernel_size=1,\
        padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        name='conv_1x1_vgg_3')
    
    # the second skip layer, 1/8
    second_skip = tf.add(second_upsample_x2, conv_1x1_vgg_3, name='second_skip')

    # upsample deconvolution by 8
    third_upsample_x8 = tf.layers.conv2d_transpose(second_skip, filters=num_classes, kernel_size=16,\
        strides=(8, 8), padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),\
        name='third_upsample_x8')
    
    return third_upsample_x8
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, [-1, num_classes], name='fcn_logits')
    correct_label_reshaped = tf.reshape(correct_label, [-1, num_classes])

    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label_reshaped, logits=logits)
    loss_op = tf.reduce_mean(cross_entropy_loss, name='fcn_loss')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name='fcn_train')

    return logits, train_op, loss_op
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_val = 0.5
    lr = 0.001
    
    def time_stamp() -> str:
        ts = time.time()
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return time_stamp
    
    for epoch in range(epochs):
        print(time_stamp() + 'Epoch {}'.format(epoch + 1))
        total_loss = 0
        for image_batch, labels_batch in get_batches_fn(batch_size):
            _, cur_loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image_batch,\
                correct_label: labels_batch, keep_prob: keep_prob_val, learning_rate: lr})
            
            total_loss += cur_loss
        
        print(time_stamp() + 'Epoch{}: loss = {:.5f}'.format(epoch + 1, total_loss))
        print('-------------------------------------------------------')
    
    print('training finished')
    

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    export_dir = './seg_model/vgg_FCN'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as session:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # placeholders
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)
        
        model_outputs = layers(layer3, layer4, layer7, num_classes)

        logits, train_op, loss_op = optimize(model_outputs, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        # initializes all of variables
        global_init = tf.global_variables_initializer()
        session.run([global_init])
        print("Model build successful, starting training.")
        epochs = 30
        batch_size = 16
        train_nn(session, epochs, batch_size, get_batches_fn, train_op, loss_op, image_input, correct_label,\
            keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, session, image_shape, logits, keep_prob, image_input)
        print("All done!")

        # save the trained segmentation model
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'image_input': image_input},\
            outputs={'output_layer': model_outputs})
        builder.add_meta_graph_and_variables(session, tags=['FCN_Udacity'],\
            signature_def_map={'predict': signature})
        builder.save()

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
