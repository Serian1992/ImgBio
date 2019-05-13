import math
import os

import cv2 as cv
import mxnet as mx
import numpy as np
from PIL import Image


from predictor import Predictor
import sys
sys.path.append('F:/Code/MyCode/CNN_Segmentation/utils')
import utils


class Tester:
    def __init__(self, config, test_shape, result_shape):
        self.config = config
        # model
        self.model_dir = config.get('model', 'model_dir')
        self.model_prefix = config.get('model', 'model_prefix')
        self.model_epoch = config.getint('model', 'model_epoch')
        self.label_num = config.getint('model', 'label_num')
        self.use_cpu = config.getboolean('model', 'use_cpu')
        self.gpus = config.get('model', 'gpus')
        if self.use_cpu is True:
                self.ctx = mx.cpu()
        else:
                self.ctx = mx.gpu(int(self.gpus))

        # data
        self.ds_rate = int(config.get('data', 'ds_rate'))
        self.cell_width = int(config.get('data', 'cell_width'))
        #self.test_shape = [int(f) for f in config.get('data', 'test_shape').split(',')]
        #self.result_shape = [int(f) for f in config.get('data', 'result_shape').split(',')]
        self.test_shape = test_shape
        self.result_shape = result_shape
        self.rgb_mean = [float(f) for f in config.get('data', 'rgb_mean').split(',')]
        # rescale for test
        self.test_scales = [float(f) for f in config.get('data', 'test_scales').split(',')]
        self.cell_shapes = [[math.ceil((l * s / self.ds_rate)*self.ds_rate) for l in self.test_shape]
                            for s in self.test_scales]
        self.modules = []
        for i, test_scale in enumerate(self.test_scales):
            predictor = mx.module.Module.load(
                prefix=os.path.join(self.model_dir, self.model_prefix),
                epoch=self.model_epoch,
                context=self.ctx)
            data_shape = (1, 3, int(self.cell_shapes[i][0]), int(self.cell_shapes[i][1]))
            predictor.bind(data_shapes=[('data', data_shape)], for_training=False)
            self.modules.append(predictor)
        self.predictor = Predictor(
            modules=self.modules,
            label_num=self.label_num,
            ds_rate=self.ds_rate,
            cell_width=self.cell_width,
            result_shape=self.result_shape,
            test_scales=self.test_scales
        )

    def preprocess(self, im):
        imgs = []
        for index, test_scale in enumerate(self.test_scales):
            # resize to test scale
            test_img = cv.resize(im, (int(im.shape[1] * test_scale), int(im.shape[0] * test_scale)),
                                 interpolation=cv.INTER_LINEAR)
            test_img = test_img.astype(np.float32)[:int(self.test_shape[0] * test_scale),
                                                   :int(self.test_shape[1] * test_scale)]
            test_img = cv.copyMakeBorder(test_img, 0, max(0, int(self.cell_shapes[index][0] * test_scale) - im.shape[0]),
                                         0, max(0, int(self.cell_shapes[index][1] * test_scale) - im.shape[1]),
                                         cv.BORDER_CONSTANT, value=self.rgb_mean)

            test_img = np.transpose(test_img, (2, 0, 1))
            # subtract rbg mean
            for i in range(3):
                test_img[i] -= self.rgb_mean[i]
            test_img = np.expand_dims(test_img, axis=0)
            mx.ndarray.array(test_img)
            imgs.append(test_img)
        return imgs


    @staticmethod
    def colorize(labels):
        
        """
        colorize the labels with predefined palette
        :param labels: labels organized in their train ids
        :return: a segmented result of colorful image as numpy array in RGB order
        """

        # label
        result_img = Image.fromarray(labels).convert('P')
        result_img.putpalette(utils.get_palette())
        return np.array(result_img.convert('RGB'))


    def predict_single(self, img, ret_softmax=False, ret_heat_map=False):
        """
        predict single image by predefined models and configuration
        :param img: image array
        :param ret_converted: whether return labels with their original ids or train ids
        :param ret_softmax: whether return softmax results
        :param ret_heat_map: whether return heat map results
        """

        rets = {}
        imgs = self.preprocess(img)
        labels = self.predictor.predict(imgs)

        # return softmax results
        if ret_softmax:
            rets['softmax'] = labels
            # feature_symbol = self.checkpoint.symbol.get_internals()['conv5_3_relu_output']
            # feature_model = mx.model.FeedForward(symbol=feature_symbol, arg_params = self.checkpoint.arg_params,
            #                                      aux_params = self.checkpoint.aux_params, ctx = self.ctx,
            #                                      allow_extra_params = True)
            # features = feature_model.predict(self.prepocess(im, 1)).squeeze()
            # np.savez(os.path.join(self.save_softmax_dir, img_name.replace('.jpg', '')), softmax=features)
        # return heat map
        if ret_heat_map:
            heat = np.max(labels, axis=0)
            heat = heat * 256 - 1
            heat_map = cv.applyColorMap(heat.astype(np.uint8), cv.COLORMAP_JET)
            rets['heat_map'] = heat_map

        results = np.argmax(labels, axis=0).astype(np.uint8)
        rets['raw'] = results
        
        return rets
            

