from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt
from ops import *
from utils import *
from scipy.io import *
import keras.backend as K



class pix2pix(object):
    def __init__(self, sess, image_size_H=256,image_size_W=256,
                 batch_size=1, sample_size=1, output_size_H=256,output_size_W=256,
                 gf_dim=128, df_dim=128, L1_lambda=200,
                 input_c_dim=3, output_c_dim=3, flow_c_dim=3, dataset_name='segmentation',
                 checkpoint_dir=None, sample_dir=None, test_dir=None, epoch=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.epoch = epoch
        self.sample_dir = sample_dir
        self.test_dir = test_dir
        self.image_size_H = image_size_H
        self.image_size_W = image_size_W
        self.sample_size = sample_size
        self.output_size_H = output_size_H
        self.output_size_W = output_size_W

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.flow_c_dim = flow_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.c_bn1 = batch_norm(name='c_bn1')
        self.c_bn2 = batch_norm(name='c_bn2')
        self.c_bn3 = batch_norm(name='c_bn3')
        self.c_bn4 = batch_norm(name='c_bn4')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_op2 = batch_norm(name='g_bn_op2')
        self.g_bn_op3 = batch_norm(name='g_bn_op3')
        self.g_bn_op4 = batch_norm(name='g_bn_op4')
        self.g_bn_op5 = batch_norm(name='g_bn_op5')
        self.g_bn_op6 = batch_norm(name='g_bn_op6')
        self.g_bn_op7 = batch_norm(name='g_bn_op7')
        self.g_bn_op8 = batch_norm(name='g_bn_op8')

        self.g_bn_opl2 = batch_norm(name='g_bn_opl2')
        self.g_bn_opl3 = batch_norm(name='g_bn_opl3')
        self.g_bn_opl4 = batch_norm(name='g_bn_opl4')
        self.g_bn_opl5 = batch_norm(name='g_bn_opl5')
        self.g_bn_opl6 = batch_norm(name='g_bn_opl6')
        self.g_bn_opl7 = batch_norm(name='g_bn_opl7')
        self.g_bn_opl8 = batch_norm(name='g_bn_opl8')

        self.g_bn_d9 = batch_norm(name='g_bn_d9')
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        self.g_bn_attention1 = batch_norm(name='g_bn_attention1')
        self.g_bn_attention2 = batch_norm(name='g_bn_attention2')
        self.g_bn_attention3 = batch_norm(name='g_bn_attention3')
        self.g_bn_attention4 = batch_norm(name='g_bn_attention4')
        self.g_bn_attention5 = batch_norm(name='g_bn_attention5')
        self.g_bn_attention6 = batch_norm(name='g_bn_attention6')
        self.g_bn_attention7 = batch_norm(name='g_bn_attention7')
        self.real_label = tf.zeros([self.batch_size, 4])

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def parser(self, record):
        features = tf.parse_single_example(record,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'img_flow': tf.FixedLenFeature([], tf.string),
                                               # 'img_flowL':tf.FixedLenFeature([], tf.string),
                                               'img_gt': tf.FixedLenFeature([], tf.string)
                                           })  # 取出包含image和label的feature对象
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        flow = tf.decode_raw(features['img_flow'], tf.uint8)
        # flowL = tf.decode_raw(features['img_flowL'], tf.uint8)
        gt = tf.decode_raw(features['img_gt'], tf.uint8)
        image = tf.reshape(image, [256, 256, 3])
        flow = tf.reshape(flow, [256, 256, 3])
        # flowL = tf.reshape(flowL, [256, 256, 1])
        gt = tf.reshape(gt, [256, 256, 1])
        label = tf.cast(features['label'], tf.int64)

        img = tf.cast(image, tf.float32) / 127.5 - 1.0
        flow = tf.cast(flow, tf.float32) / 127.5 - 1.0
        # flowL = tf.cast(flowL, tf.float32) / 127.5 - 1.0
        gt = tf.cast(gt, tf.float32) / 255.
        return img, gt, flow,  label

    def build_model(self):

        # filename = ["CDnet_train_Shadow_aug.tfrecords"]#"CDnet_train_Shadow.tfrecords"
        filename = ['CDnet_train_merge3OP_200627.tfrecords']
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser)

        dataset = dataset.shuffle(buffer_size=500,seed=12138)
        print("batch size:", self.batch_size)
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(batch_size=self.batch_size)

        print("DATASET", dataset)

        iterator = dataset.make_initializable_iterator()
        img, gt, flow,  label = iterator.get_next()

        # img = img.reshape((self.batch_size, 256, 256, 3))
        # gt = gt.reshape((self.batch_size, 256, 256, 1))
        # flow = flow.reshape((self.batch_size, 256, 256, 1))
        print("ITERATOR", iterator)

        self.sess.run(iterator.initializer)

        self.N_GPUs = 2

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        tower_grads_g = []

        self.lr = 0.0005
        # g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.9)  # 0.00002
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.95)
        with tf.variable_scope(tf.get_variable_scope()):
            for ngpu in range(self.N_GPUs):
                with tf.device('/gpu:%d' % (ngpu + 1)):
                    with tf.name_scope('GPU_%d' % ngpu) as scope:

                        self.gloss1,self.gloss2 = self.get_loss(img, gt, flow,  label, scope)
                        self.g_loss = self.gloss1 + self.gloss2

                        t_vars = tf.trainable_variables()
                        self.g_vars = [var for var in t_vars if 'g_' in var.name]
                        tf.get_variable_scope().reuse_variables()
                        grad_G = g_optim.compute_gradients(self.g_loss, var_list=self.g_vars)
                        tower_grads_g.append(grad_G)

        grad_G_ave = self.average_gradients(tower_grads_g)
        self.trainG = g_optim.apply_gradients(grad_G_ave, global_step=global_step)

        print("Plot Initialization")

        # self.d_sum = tf.summary.histogram("d", self.D)
        # self.d__sum = tf.summary.histogram("d_", self.D_)
        self.real_A_sum = tf.summary.image("real_A", self.real_A)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_C_sum = tf.summary.image("real_flow", self.real_C)
        self.g1_loss_sum = tf.summary.scalar("g1_loss", self.gloss1)
        self.g2_loss_sum = tf.summary.scalar("g2_loss", self.gloss2)
        # self.real_D_sum = tf.summary.image("real_flowL",self.real_D)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        # self.dice_loss_sum = tf.summary.scalar('dice_loss',(1-self.th) * 400)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)
        self.saver = tf.train.Saver()

    def train(self,args):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            init = tf.global_variables_initializer()
            self.sess.run(init)

        sample_files = glob('/data/tmpu1/WZQ/Test/dataset/dataset2/train/*.png')
        # sample_files = glob('/data/tmpu1/WZQ/Test/dataset/CD_split/Shadow/test/*.jpg')
        # sample_files = glob('./datasets/{}/test/*.jpg'.format(self.dataset_name))
        n = [int(i) for i in map(lambda x: x.split('/train_')[-1].split('.png')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
        self.g_sum = tf.summary.merge(
            [self.real_A_sum, self.fake_B_sum, self.real_B_sum, self.real_C_sum, self.g_loss_sum])
        # self.dice_sum = tf.summary.merge([self.dice_loss_sum])
        self.g1_sum = tf.summary.merge([self.g1_loss_sum])
        self.g2_sum = tf.summary.merge([self.g2_loss_sum])


        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
        print("training")
        start_time = time.time()
        for step in xrange(args.epoch):
        # for step in xrange(self.epoch):

            #change
            # if step in []:
            #     self.lr = 0.1 * self.lr
            # if step == 5000 or step == 7000:
            #     self.lr = 0.1 * self.lr

            start_tmp = time.time()
            _, summary_str = self.sess.run([self.trainG, self.g_sum])
            self.writer.add_summary(summary_str, step)

            _, summary_str = self.sess.run([self.trainG, self.g_sum])
            self.writer.add_summary(summary_str, step)

            # _, summary_str = self.sess.run([self.trainG, self.dice_sum])
            # self.writer.add_summary(summary_str, step)

            _, summary_str = self.sess.run([self.trainG, self.g1_sum])
            self.writer.add_summary(summary_str, step)

            _, summary_str = self.sess.run([self.trainG, self.g2_sum])
            self.writer.add_summary(summary_str, step)

            duration = time.time() - start_tmp
            # print("eopch:%d, duration: %.4f" %(step, duration))
            if step != 0 and step % 100 == 0:
                num_example_per_step = self.batch_size * self.N_GPUs
                example_per_sec = num_example_per_step / duration
                sec_per_batch = duration / self.N_GPUs
                print("step: %d, example_per_sec: %.4f, sec_per_batch: %.4f ,g_loss: %.4f,g1_loss: %.8f,g2_loss: %.8f"
                      % (step, example_per_sec, sec_per_batch, self.g_loss.eval(), self.gloss1.eval(), self.gloss2.eval()))

                ranint = np.random.randint(low=0, high=len(sample_files) - self.batch_size)

                data = sample_files[ranint:ranint + self.batch_size]
                # np.random.choice(glob('./datasets/{}/test/*.jpg'.format(self.dataset_name)),self.batch_size)
                sample = [load_data(sample_file) for sample_file in data]

                sample_image1 = np.array(sample).astype(np.float32)

                S, B, A, C = self.sess.run(
                    [self.fake_B_sample, self.real_data_B, self.real_data_A, self.real_data_C],
                    feed_dict={self.real_data_ABC: sample_image1})
                # B1 = np.tile(B1, [1, 1, 1, 3])
                S = np.tile(S, [1, 1, 1, 3])
                # C = np.tile(C, [1, 1, 1, 3])
                # D = np.tile(D, [1, 1, 1, 3])
                S = S * 2.0 - 1.0
                tmp = np.concatenate((A, B), axis=2)
                tmp = np.concatenate((tmp, C), axis=2)
                # tmp = np.concatenate((tmp, D), axis=2)
                tmp = np.concatenate((tmp, S), axis=2)
                save_images(tmp, [self.batch_size, 1],
                            './{}/train_{:02d}_{:04d}.jpg'.format(self.sample_dir, step, 1))

            # print("step: %d time: %.4f " % (step, time.time() - start_time,))
            if np.mod(step + 1, 500) == 0:
                self.save(self.checkpoint_dir, step)
        self.test(args)
        self.sess.close()

    def get_loss1(self, img, gt, flow, label, scope):

        self.inference(img, gt, flow)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_B, logits=self.fake_B_logits))
        #g_loss = self.L1_lambda * tf.reduce_mean(tf.abs(self.fake_B - self.real_B))

        return g_loss

    def get_loss(self, img, gt, flow, label, scope):

        self.inference(img, gt, flow)
        # get pixel > 200.0 / 255.0
        self.real_B = get_fg0(self.real_B)

        g_loss_1 = 0.2 * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.th = dice_coef(self.real_B, self.fake_B, smooth=0.0)
        if tf.greater(self.th, tf.constant(value=0.95, dtype='float32')) is True:
            # beta = get_beta(self.real_B)
            beta = get_area(self.real_B) / 4.0
            g_loss_2 = 0.8 * tf.reduce_mean(
                sigmoid_focal_crossentropy_new(y_true=self.real_B, y_pred=self.fake_B,
                                               alpha=0.75, gamma=0.0, beta=beta, from_logits=False))
        else:
            g_loss_2 = 0.8 * tf.reduce_mean(
                sigmoid_focal_crossentropy(y_true=self.real_B, y_pred=self.fake_B,
                                           alpha=0.75, gamma=0.0, from_logits=False))

        return g_loss_1, g_loss_2


    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.


        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    def inference(self, img, gt, flow):

        tf.reshape(img, [self.batch_size, 256, 256, 3])
        tf.reshape(gt, [self.batch_size, 256, 256, 1])
        tf.reshape(flow, [self.batch_size, 256, 256, 3])
        # tf.reshape(flowL, [self.batch_size, 256, 256, 1])


        self.real_A = img
        self.real_B = gt
        self.real_C = flow
        # self.real_D = flowL

        self.real_data_ABC = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size_H, self.image_size_W,
                                         self.input_c_dim * 3],
                                        name='real_data_ABC')

        self.real_data_A = self.real_data_ABC[:, :, :, :self.input_c_dim]
        self.real_data_B = self.real_data_ABC[:, :, :, self.input_c_dim * 2:self.input_c_dim * 3]
        self.real_data_C = self.real_data_ABC[:, :, :, self.input_c_dim:self.input_c_dim * 2]
        # self.real_data_D = self.real_data_ABCD[:, :, :, self.input_c_dim * 2:self.input_c_dim * 3]

        # self.real_data_C = self.real_data_C[:, :, :, 0]
        # self.real_data_C = self.real_data_C[:, :, :, np.newaxis]
        # self.real_data_D = self.real_data_D[:, :, :, 0]
        # self.real_data_D = self.real_data_D[:, :, :, np.newaxis]

        self.real_AC = tf.concat([self.real_A, self.real_C], 3)


        with tf.variable_scope("generator"):
            self.fake_B, self.fake_B_logits = self.generator(image=img, optical=self.real_C,) #tf.tile(self.real_C, [1, 1, 1, 3])
        with tf.variable_scope("generator", reuse=True):
            self.fake_B_sample, self.fake_B_logits_sample = self.generator(self.real_data_A, self.real_data_C) #tf.tile(self.real_data_C, [1, 1, 1, 3])
        self.real_ABC = tf.concat([self.real_AC, self.real_B], 3)#self.real_AC
        self.fake_ABC = tf.concat([self.real_AC, self.fake_B], 3)#self.real_AC

    def generator(self, image, optical,  y=None):
        s = self.output_size_H
        s2, s4, s8, s16, s32, s64, s128, s256 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
            s / 64), int(s / 128), int(s / 256)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, k_h=7, k_w=7, name='g_e1_conv')
        op1 = conv2d(optical, self.gf_dim, k_h=7, k_w=7, name='g_op1_conv')

        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, k_h=7, k_w=7, name='g_e2_conv'))
        op2 = self.g_bn_op2(conv2d(lrelu(op1), self.gf_dim * 2, k_h=7, k_w=7, name='g_op2_conv'))

        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
        op3 = self.g_bn_op3(conv2d(lrelu(op2), self.gf_dim * 4, name='g_op3_conv'))

        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
        op4 = self.g_bn_op4(conv2d(lrelu(op3), self.gf_dim * 8, name='g_op4_conv'))

        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
        op5 = self.g_bn_op5(conv2d(lrelu(op4), self.gf_dim * 8, name='g_op5_conv'))

        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
        op6 = self.g_bn_op6(conv2d(lrelu(op5), self.gf_dim * 8, name='g_op6_conv'))

        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, k_h=3, k_w=3, name='g_e7_conv'))
        op7 = self.g_bn_op7(conv2d(lrelu(op6), self.gf_dim * 8, k_h=3, k_w=3, name='g_op7_conv'))

        # e7 is (2 x 2 x self.gf_dim*16)
        e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 16, k_h=3, k_w=3, name='g_e8_conv'))
        op8 = self.g_bn_op8(conv2d(lrelu(op7), self.gf_dim * 16, k_h=3, k_w=3, name='g_op8_conv'))

        fusion0 = tf.concat([e8, op8], 3)
        fusion0 = conv2d(lrelu(fusion0), self.gf_dim * 16, k_h=3, k_w=3, name='fusion0')

        d1, d1_w, d1_b = deconv2d(lrelu(fusion0),
                                  [self.batch_size, s128, s128, self.gf_dim * 8], k_h=3, k_w=3, name='g_d1',
                                  with_w=True)
        attention1 = self.g_bn_attention1(
            conv2d(lrelu(d1), self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1, name='g_attention1'))
        d1 = tf.nn.dropout(self.g_bn_d1(d1), rate=0.5,seed=123)
        # d1 = tf.nn.dropout(self.g_bn_d1(d1), rate=0.4,seed=123)

        fusion1 = tf.concat([e7, op7], 3)
        fusion1 = conv2d(lrelu(fusion1), self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1, name='fusion1')

        att1 = tf.multiply(tf.nn.sigmoid(attention1), fusion1)
        d1 = tf.concat([d1, att1], 3)  # d1 + self.att1 #d1 + self.att1
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(lrelu(d1),
                                  [self.batch_size, s64, s64, self.gf_dim * 8], k_h=3, k_w=3, name='g_d2', with_w=True)

        attention2 = self.g_bn_attention2(
            conv2d(lrelu(d2), self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1, name='g_attention2'))
        d2 = tf.nn.dropout(self.g_bn_d2(d2), rate=0.5,seed=223)
        # d2 = tf.nn.dropout(self.g_bn_d2(d2), rate=0.4,seed=223)

        fusion2 = tf.concat([e6, op6], 3)
        fusion2 = conv2d(lrelu(fusion2), self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1, name='fusion2')

        att2 = tf.multiply(tf.nn.sigmoid(attention2), fusion2)
        d2 = tf.concat([d2, att2], 3)  # d2 + self.att2#d2 + self.att2
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(lrelu(d2),
                                  [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3', with_w=True)

        attention3 = self.g_bn_attention3(conv2d(lrelu(d3), self.gf_dim * 8, d_h=1, d_w=1, name='g_attention3'))
        d3 = tf.nn.dropout(self.g_bn_d3(d3), rate=0.5,seed=323)
        # d3 = tf.nn.dropout(self.g_bn_d3(d3), rate=0.4,seed=323)

        fusion3 = tf.concat([e5, op5], 3)
        fusion3 = conv2d(lrelu(fusion3), self.gf_dim * 8, d_h=1, d_w=1, name='fusion3')

        att3 = tf.multiply(tf.nn.sigmoid(attention3), fusion3)
        d3 = tf.concat([d3, att3], 3)  # d3 + self.att3#d3 + self.att3

        d4, d4_w, d4_b = deconv2d(lrelu(d3),
                                  [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4', with_w=True)

        attention4 = self.g_bn_attention4(conv2d(lrelu(d4), self.gf_dim * 8, d_h=1, d_w=1, name='g_attention4'))
        d4 = self.g_bn_d4(d4)

        fusion4 = tf.concat([e4, op4], 3)
        fusion4 = conv2d(lrelu(fusion4), self.gf_dim * 8, d_h=1, d_w=1, name='fusion4')

        att4 = tf.multiply(tf.nn.sigmoid(attention4), fusion4)
        d4 = tf.concat([d4, att4], 3)  # d4 + self.att4#d4 + self.att4

        d5, d5_w, d5_b = deconv2d(lrelu(d4),
                                  [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5', with_w=True)

        attention5 = self.g_bn_attention5(conv2d(lrelu(d5), self.gf_dim * 4, d_h=1, d_w=1, name='g_attention5'))
        d5 = self.g_bn_d5(d5)

        fusion5 = tf.concat([e3, op3], 3)
        fusion5 = conv2d(lrelu(fusion5), self.gf_dim * 4, d_h=1, d_w=1, name='fusion5')

        att5 = tf.multiply(tf.nn.sigmoid(attention5), fusion5)
        d5 = tf.concat([d5, att5], 3)  # d5 + self.att5#d5 + self.att5

        d6, d6_w, d6_b = deconv2d(lrelu(d5),
                                  [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6', with_w=True)

        attention6 = self.g_bn_attention6(conv2d(lrelu(d6), self.gf_dim * 2, d_h=1, d_w=1, name='g_attention6'))
        d6 = self.g_bn_d6(d6)

        fusion6 = tf.concat([e2, op2], 3)
        fusion6 = conv2d(lrelu(fusion6), self.gf_dim * 2, d_h=1, d_w=1, name='fusion6')

        att6 = tf.multiply(tf.nn.sigmoid(attention6), fusion6)
        d6 = tf.concat([d6, att6], 3)  # d6 + self.att6#d6 + self.att6

        d7, d7_w, d7_b = deconv2d(lrelu(d6),
                                  [self.batch_size, s2, s2, self.gf_dim], k_h=7, k_w=7, name='g_d7', with_w=True)

        attention7 = self.g_bn_attention7(
            conv2d(lrelu(d7), self.gf_dim, k_h=7, k_w=7, d_h=1, d_w=1, name='g_attention7'))
        d7 = self.g_bn_d7(d7)

        fusion7 = tf.concat([e1, op1], 3)
        fusion7 = conv2d(lrelu(fusion7), self.gf_dim, k_h=7, k_w=7, d_h=1, d_w=1, name='fusion7')

        self.att7 = tf.multiply(tf.nn.sigmoid(attention7), fusion7)
        d7 = tf.concat([d7, self.att7], 3)  # d7 + self.att7#d7 + self.att7

        d8, d8_w, d8_b = deconv2d(lrelu(d7),
                                  [self.batch_size, s, s, 1], k_h=7, k_w=7, name='g_d8', with_w=True)
        return tf.nn.sigmoid(d8), d8

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s" % (self.dataset_name, self.output_size_H)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.output_size_H)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    def test(self,args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files1 = glob('/data/tmpu1/WZQ/Test/dataset/dataset2/test/*.png')
        # sample_files1 = glob('./datasets/{}/test/*.jpg'.format(self.dataset_name))
        # sample_files1 = glob('/data/tmpu1/WZQ/Test/dataset/CD_split/Shadow/test/*.jpg')
        # n = [int(i) for i in map(lambda x: x.split('/train_')[-1].split('.png')[0], sample_files1)]
        n = [int(i) for i in map(lambda x: x.split('/test_')[-1].split('.png')[0], sample_files1)]
        sample_files1 = [x for (y, x) in sorted(zip(n, sample_files1))]

        time_start = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            # load testing input
        time_start = time.time()
        print("Loading testing images ...")
        for idx in xrange(0, len(sample_files1), self.batch_size):

            data1 = sample_files1[idx: idx + self.batch_size]
            sample1 = [load_data(sample_file) for sample_file in data1]

            sample_image1 = np.array(sample1).astype(np.float32)

            S1, B1, A1, C1 = self.sess.run([self.fake_B_sample, self.real_data_B, self.real_data_A, self.real_data_C],
                                           feed_dict={self.real_data_ABC: sample_image1})
            #B1 = np.tile(B1, [1, 1, 1, 3])
            S1 = np.tile(S1, [1, 1, 1, 3])
            S1 = S1 * 2.0 - 1.0
            # C1 = np.tile(C1, [1, 1, 1, 3])
            # D1 = np.tile(D1, [1, 1, 1, 3])
            tmp1 = np.concatenate((A1, B1), axis=2)
            tmp1 = np.concatenate((tmp1, C1), axis=2)
            # tmp1 = np.concatenate((tmp1, D1), axis=2)
            tmp1 = np.concatenate((tmp1, S1), axis=2)
            save_images(tmp1, [self.batch_size, 1],
                            './{}/test_{:04d}.png'.format(args.test_dir, idx + 1))
            print("testing images:", idx)
        time_end = time.time()
        print('time cost:', time_end - time_start)

    def demo(self,args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        path1 = './dataset/demo_data/'

        sample_files1 = glob(path1+'*.png')
        # sample_files1 = glob('/data/tmpu1/WZQ/Test/dataset/LIMU/Intersection/merge/*.png')
        n = [int(i) for i in map(lambda x: x.split('/test_')[-1].split('.png')[0], sample_files1)]
        sample_files1 = [x for (y, x) in sorted(zip(n, sample_files1))]

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            # load testing input
        print("Loading demo images ...")
        for idx in xrange(0, len(sample_files1), 1):

            data1 = sample_files1[idx: idx + 1]
            # print(data1)
            name = data1[0].split('/')[-1]
            sample1 = [load_data1OP(sample_file) for sample_file in data1]

            sample_image1 = np.array(sample1).astype(np.float32)

            S1, B1, A1, C1 = self.sess.run([self.fake_B_sample, self.real_data_B, self.real_data_A, self.real_data_C],
                                           feed_dict={self.real_data_ABC: sample_image1})

            S1 = np.tile(S1, [1, 1, 1, 3])
            S1 = S1 * 2.0 - 1.0
            save_images(S1, [1, 1], './dataset/demo_result/'+name)