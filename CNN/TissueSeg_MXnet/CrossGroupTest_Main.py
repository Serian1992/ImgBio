import argparse
import ConfigParser
import os
import sys
import time
from PIL import Image

import cv2 as cv
import numpy as np

print os.getcwd()
os.chdir("F:\Code\MyCode\CNN_Segmentation")

sys.path.append('./utils')
from tester import Tester


class ImageListTester:
    def __init__(self, config):
        self.config = config
        # # model
        self.model_dir = config.get('model', 'model_dir')
        self.model_prefix = config.get('model', 'model_prefix')
        self.model_epoch = config.getint('model', 'model_epoch')
        self.result_dir = config.get('model', 'result_dir')
        #if not os.path.isdir(self.result_dir):
         #   os.mkdir(self.result_dir)
        #if not os.path.isdir(os.path.join(self.result_dir, 'visualization')):
         #   os.mkdir(os.path.join(self.result_dir, 'visualization'))
        if not os.path.isdir(os.path.join(self.result_dir, 'score')):
            os.mkdir(os.path.join(self.result_dir, 'score'))

        # data
        self.image_list = config.get('data', 'image_list')
        self.test_img_dir = config.get('data', 'test_img_dir')
        self.result_shape = [int(f) for f in config.get('data', 'result_shape').split(',')]
        self.test_shape = [int(f) for f in config.get('data', 'test_shape').split(',')]
        # initialize tester
        self.tester = Tester(self.config, self.test_shape, self.result_shape)

    def predict_single(self, item):
        # img_name = item.strip().replace('/', '_')   
        img_path = os.path.join(self.test_img_dir, item.strip().split('\t')[1])
        img_name = img_path.strip().split('/')[-1]

        # read image as rgb
        im = cv.imread(img_path)[:, :, ::-1]
        result_width = self.result_shape[1]
        result_height = self.result_shape[0]

        #concat_img = Image.new('RGB', (result_width * 2, result_height * 2))

        results = self.tester.predict_single(
            img=im,
            ret_heat_map=True,
            ret_softmax=True)

        # label
        heat_map = results['heat_map']
        raw_labels = results['raw'] + 1
        softmax = results['softmax']

        confidence = float(np.max(softmax, axis=0).mean())

        #result_img = Image.fromarray(self.tester.colorize(raw_labels)).resize(self.result_shape[::-1])

        # paste raw image
        #concat_img.paste(Image.fromarray(im).convert('RGB'), (0, 0))
        # paste color result
        #concat_img.paste(result_img, (0, result_height))
        # paste blended result
        #concat_img.paste(Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0)),
                         #(result_width, 0))
        # paste heat map
        #concat_img.paste(Image.fromarray(heat_map[:, :, [2, 1, 0]]).resize(self.result_shape[::-1]),
                         #(result_width, result_height))
        #concat_img.save(os.path.join(self.result_dir, 'visualization', img_name.replace('jpg', 'png')))

        # save results for score
        cv.imwrite(os.path.join(self.result_dir, 'score', img_name.replace('jpg', 'png')), raw_labels)
        return confidence, img_path
        

    def predict_all(self):
        img_list = [line for line in open(self.image_list, 'r')]
        idx = 0
        conf_lst = []
        for item in img_list[:]:
            idx += 1
            start_time = time.time()
            item = item.replace('\\','/')
            conf_lst.append(self.predict_single(item)) 
            
            print 'Process %d out of %d image ... %s, time cost:%.3f, confidence:%.3f' % \
                  (idx, len(img_list), item.strip().split('/')[-1].replace('mask','img').replace('png','jpg'), time.time() - start_time, conf_lst[-1][0])
        conf_file = open(os.path.join(self.result_dir, self.model_prefix + str(self.model_epoch) + '.txt'), 'w')
        conf_lst.sort()
        for item in conf_lst:

            print >> conf_file, "{}\t{}".format(item[1], item[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", help="display the current running experiment", type=int, default=1)
    
    parser.add_argument("--model_dir", help="display the model_dir", default='./Models/crossEx/Folder1/TusimpleDUC_Seg/2018_02_13_23_18_29')
    parser.add_argument("--model_prefix", help="display the model_prefix", default='TusimpleDUC_Seg')
    parser.add_argument("--model_epoch", help="display the model_epoch", type=int, default=30)
    parser.add_argument("--result_dir", help="display the result_dir", default='./Models/STusimpleDUC/Valresults/')
    parser.add_argument("--gpus", help="display the gpus", default='0')
    parser.add_argument("--label_num", help="display the label_num", type=int, default=3)
    parser.add_argument("--image_list", help="display the image_list", default='./Data/TMAD/Folder1/val_FullImage_list.lst')
    parser.add_argument("--test_img_dir", help="display the test_img_dir", default='./Data/TMAD/Folder1/Val')
    parser.add_argument("--cell_width", help="display the cell_width", type=int, default=8)
    
    args = parser.parse_args()

    config_path = './Configs/test_full_image_TusimpleDUC_TCGATissue.cfg'
    config = ConfigParser.RawConfigParser()
    config.read(config_path)
    
    config.set('model','model_dir',value=args.model_dir)
    config.set('model','model_prefix',value=args.model_prefix)
    config.set('model','model_epoch',value=args.model_epoch)
    config.set('model','result_dir',value=args.result_dir)
    config.set('model','gpus',value=args.gpus)
    config.set('model','label_num',value=args.label_num)
    config.set('data','image_list',value=args.image_list)
    config.set('data','test_img_dir',value=args.test_img_dir)
    config.set('data','gt_dir',value=args.test_img_dir)
    config.set('data','cell_width',value=args.cell_width)
    
    tester = ImageListTester(config)
    tester.predict_all()
