# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:16:16 2017

@author: luzix
"""
import mxnet as mx
import logging
import sys

def get_Init_model(InitializedModelName='FCN', ctx=mx.cpu(), cell_cap=2, label_num=3, data_shape=(1,3,160,160), ignore_label=0, bn_use_global_stats=True, aspp_num=4, aspp_stride=6, InitType=1):
    args = None
    auxs = None
    
    # Initialize FCN
    if InitializedModelName == 'FCN':
        sys.path.append('./Models/FCN')
        import FCNInit_symbol_resnet
        import FCNInit_params
        symbol = FCNInit_symbol_resnet.get_fcn8s_symbol(num_classes=label_num, workspace=1024, ignore_label=ignore_label)
        if InitType==1:
           # _,pretrained_args, pretrained_auxs = mx.model.load_checkpoint('./Models/FCN/VGG_FC_ILSVRC_16_layers',74)
            _,pretrained_args, pretrained_auxs = mx.model.load_checkpoint('./Models/TusimpleDUC/init',0)
            args, auxs = FCNInit_params.init_from_VGG16(ctx, symbol, pretrained_args, pretrained_auxs, data_shape)
        
    
    # DeepLabV2
    elif InitializedModelName == 'DeepLabV2':
        
        symbol = FCNInit_symbol.get_fcn8s_symbol(num_classes=21, workspace=1024)
        _, pretrained_args, pretrained_auxs = mx.model.load_checkpoint('VGG_FC_ILSVRC_16_layers',74)
        args, auxs = FCNInit_params.init_from_VGG16(ctx, symbol, pretrained_args, pretrained_auxs)
    
    # Tusimple-DUC
    elif InitializedModelName == 'Tusimple-DUC':
        sys.path.append('./Models/TusimpleDUC')
        import TusimpleDUCInit_symbol
        import TusimpleDUCInit_symbol2
        import TusimpleDUCInit_symbol3
        import TusimpleDUCInit_symbol4
        import TusimpleDUCInit_params
        if InitType==1:
            symbol = TusimpleDUCInit_symbol.get_symbol_duc_hdc(cell_cap=cell_cap, label_num=label_num, ignore_label=ignore_label, bn_use_global_stats=bn_use_global_stats, aspp_num=aspp_num, aspp_stride=aspp_stride)
        elif InitType==2:    
            symbol = TusimpleDUCInit_symbol2.get_symbol_duc_hdc(cell_cap=cell_cap, label_num=label_num, ignore_label=ignore_label, bn_use_global_stats=bn_use_global_stats, aspp_num=aspp_num, aspp_stride=aspp_stride)
        elif InitType==3:
            symbol = TusimpleDUCInit_symbol3.get_symbol_duc_hdc(cell_cap=cell_cap, label_num=label_num, ignore_label=ignore_label, bn_use_global_stats=bn_use_global_stats, aspp_num=aspp_num, aspp_stride=aspp_stride)
        elif InitType==4:
            symbol = TusimpleDUCInit_symbol4.get_symbol_duc_hdc(cell_cap=cell_cap, label_num=label_num, ignore_label=ignore_label, bn_use_global_stats=bn_use_global_stats, aspp_num=aspp_num, aspp_stride=aspp_stride)
            
        if InitType>0:
            _,pretrained_args, pretrained_auxs = mx.model.load_checkpoint('./Models/TusimpleDUC/init',0)
            args, auxs = TusimpleDUCInit_params.init_from_DeeplabV2(ctx, symbol, pretrained_args, pretrained_auxs, data_shape)
        
    
    # STusimple-DUC  
    elif InitializedModelName == 'STusimple-DUC':
        sys.path.append('./Models/STusimpleDUC')
        import STusimpleDUCInit_symbol3
        import STusimpleDUCInit_params3
        symbol = STusimpleDUCInit_symbol3.get_symbol_stacked_duc_hdc(cell_cap=cell_cap, label_num=label_num, ignore_label=ignore_label, bn_use_global_stats=bn_use_global_stats, aspp_num=aspp_num, aspp_stride=aspp_stride, datashape=data_shape)
        _,pretrained_args, pretrained_auxs = mx.model.load_checkpoint('./Models/TusimpleDUC/TusimpleDUC_Seg/2018_01_07_14_03_43/TusimpleDUC_Seg',50)
        args, auxs = STusimpleDUCInit_params3.init_from_DeeplabV2(ctx, symbol, pretrained_args, pretrained_auxs, data_shape)
        #symbol, args, auxs = mx.model.load_checkpoint('./Models/STusimpleDUC/STusimpleDUC_Seg/2018_02_07_13_49_58/STusimpleDUC_Seg',27)
        
        
    # SCNN
    elif InitializedModelName == 'SCNN':
        sys.path.append('./Models/StackedCNN')
        import SCNNInit_symbol
        import SCNNInit_params
        s,pretrained_args,pretrained_auxs=mx.model.load_checkpoint('./Models/StackedCNN/densenet-161',0)
        symbol = SCNNInit_symbol.get_SCNN_M1(s, label_num=label_num, ignore_label=ignore_label, supervisionLevel=2, name='M1')
        args, auxs = SCNNInit_params.init_from_DenseNet161(ctx, symbol, pretrained_args, pretrained_auxs, data_shape)
        
        
    else :
        logging.error('Unknown Model Type!')
        
    return symbol, args, auxs

if __name__ == '__main__':
    get_Init_model('SCNN', mx.gpu(0), 2, 3, data_shape=(1,3,224,224), ignore_label=3, bn_use_global_stats=True, aspp_num=4, aspp_stride=6, InitType=1)
    
        
        
        
    
    
    