#!/usr/bin/env python
# ref: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna,
#      "Rethinking the Inception Architecture for Computer Vision"
#      https://arxiv.org/abs/1512.00567
# see also https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py


import numpy

from layers import *

#------------------------------------------------------------------------------

def addInceptionT1(model, name , bottom,
                   p1_num_out,  p2_1x1, p2_num_out, p3_1x1, p3_num_out, p4_num_out, pool):
    # 1x1
    model, top1 = addConvBnRelu(model, name='{}/p1_1x1'.format(name), bottom=bottom, num_output=p1_num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    # 1x1->3x3
    model, top2 = addConvBnRelu(model, name='{}/p2_1x1'.format(name), bottom=bottom, num_output=p2_1x1,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_3x3'.format(name), bottom=top2,   num_output=p2_num_out,
                                kernel_size=3, stride=1, pad=1, filler="xavier")
    # "5x5 factorized into 1x1->3x3->3x3
    model, top3 = addConvBnRelu(model, name='{}/p3_1x1'.format(name), bottom=bottom, num_output=p3_1x1,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_3x3a'.format(name), bottom=top3,  num_output=p3_num_out,
                                kernel_size=3, stride=1, pad=1, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_3x3b'.format(name), bottom=top3,  num_output=p3_num_out,
                                kernel_size=3, stride=1, pad=1, filler="xavier")
    # pool->[1x1]
    model, top4 = addPool(model, name='{}/p4_pool'.format(name), bottom=bottom,
                                kernel_size=3, stride=1, pad=1, pool_type=pool)
    model, top4 = addConvBnRelu(model, name='{}/p4_1x1'.format(name), bottom=top4, num_output=p4_num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    # concat
    tops = [top1, top2, top3, top4]
    model, top = addConcat(model, name='{}/concat'.format(name), bottoms=tops)
    return model, top

#------------------------------------------------------------------------------
def addReductionR1(model, name , bottom,  p1_1x1, p1_3x3, p1_3x3s2, p2_3x3s2, pool):

    # p1:1x1->[3x3]->[3x3/2]
    model, top1 = addConvBnRelu(model, name='{}/p1_1x1'.format(name),  bottom=bottom, num_output=p1_1x1,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top1 = addConvBnRelu(model, name='{}/p1_3x3a'.format(name), bottom=top1,   num_output=p1_3x3,
                                kernel_size=3, stride=1, pad=1, filler="xavier")
    model, top1 = addConvBnRelu(model,name='{}/p1_3x3b'.format(name),  bottom=top1,   num_output=p1_3x3s2,
                                kernel_size=3, stride=2, pad=0, filler="xavier")

    # p2:1x1->[3x3]->[3x3/2]
#   top2 = addConvBnRelu(model, name='{}/p2_1x1'.format(name), bottom=bottom, num_output=p2_1x1, kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_3x3'.format(name), bottom=bottom,  num_output=p2_3x3s2,
                                kernel_size=3, stride=2, pad=0, filler="xavier")

    # p3: pool stride 2
    model, top3 = addPool(model, name='{}/p3_pool'.format(name), bottom=bottom,
                          kernel_size=3, stride=2, pad=0, pool_type=pool)

    # concat
    tops = [top1, top2, top3]
    model, top = addConcat(model,  name='{}/concat'.format(name), bottoms=tops)
    return model, top

#------------------------------------------------------------------------------
def addInceptionT2(model, name , bottom,  num_proj, num_out, pool):
    # p1:1x1
    model, top1 = addConvBnRelu(model, name='{}/p1_1x1'.format(name), bottom=bottom, num_output=num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    # p2:1x1->1x7->7x1
    model, top2 = addConvBnRelu(model, name='{}/p2_1x1'.format(name), bottom=bottom, num_output=num_proj,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_1x7'.format(name), bottom=top2, num_output=num_proj,
                                kernel_h=1, kernel_w=7, pad_h=0, pad_w=3, stride=1, filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_7x1'.format(name), bottom=top2, num_output=num_out,
                                kernel_h=7, kernel_w=1, pad_h=3, pad_w=0, stride=1, filler="xavier")

    # p3: 1x1->1x7->7x1->1x7->7x1
    model, top3 = addConvBnRelu(model, name='{}/p3_1x1'.format(name), bottom=bottom, num_output=num_proj,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_1x7a'.format(name), bottom=top3, num_output=num_proj,
                                kernel_h=1, kernel_w=7, pad_h=0, pad_w=3, stride=1, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_7x1a'.format(name), bottom=top3, num_output=num_proj,
                                kernel_h=7, kernel_w=1, pad_h=3, pad_w=0, stride=1, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_1x7b'.format(name), bottom=top3, num_output=num_proj,
                                kernel_h=1, kernel_w=7, pad_h=0, pad_w=3, stride=1, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_7x1b'.format(name), bottom=top3, num_output=num_out,
                                kernel_h=7, kernel_w=1, pad_h=3, pad_w=0, stride=1, filler="xavier")

    # pool->[1x1]
    model, top4 = addPool(model, name='{}/p4_pool'.format(name), bottom=bottom,
                                kernel_size=3, stride=1, pad=1, pool_type=pool)
    model, top4 = addConvBnRelu(model, name='{}/p4_1x1'.format(name), bottom=top4, num_output=num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    # concat
    tops = [top1, top2, top3, top4]
    model, top = addConcat(model, name='{}/concat'.format(name), bottoms=tops)
    return model, top

#------------------------------------------------------------------------------
    # Reduction block 2 -------------------------------------------------------
    # 192x[1x1]->320x[3x3/2] + 192x[1x1]->192x[1x7]->192x[7x1]->192x[3x3/2] + [MAX_3x3/2]

def addReductionR2(model, name , bottom,  num_proj, num_out, pool):

    # p1:[1x1]->[3x3/2]
    model, top1 = addConvBnRelu(model, name='{}/p1_1x1'.format(name), bottom=bottom, num_output=num_proj,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top1 = addConvBnRelu(model,name='{}/p1_3x3'.format(name), bottom=top1,  num_output=num_out,
                                kernel_size=3, stride=2, pad=0, filler="xavier")

    # p2:[1x1]->[1x7]->[7x1]->[3x3]s2
    model, top2 = addConvBnRelu(model, name='{}/p2_1x1'.format(name), bottom=bottom, num_output=num_proj,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_1x7'.format(name), bottom=top2, num_output=num_proj,
                                kernel_h=1, kernel_w=7, pad_h=0, pad_w=3, stride=1,filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_7x1'.format(name), bottom=top2, num_output=num_proj,
                                kernel_h=7, kernel_w=1, pad_h=3, pad_w=0, stride=1,filler="xavier")
    model, top2 = addConvBnRelu(model, name='{}/p2_3x3'.format(name), bottom=top2, num_output=num_proj,
                                kernel_size=3, stride=2, pad=0, filler="xavier")
    # p3: pool_3x3/2
    model, top3 = addPool(model, name='{}/p3_pool'.format(name), bottom=bottom,
                                kernel_size=3, stride=2, pad=0, pool_type=pool)
    # concat
    tops = [top1, top2, top3]
    model, top = addConcat(model,  name='{}/concat'.format(name), bottoms=tops)
    return model, top

#------------------------------------------------------------------------------
# 35 x 35 x 192:  Inception block T3
#     [1x1]   [1x1]->[1x3]+[3x1]   [1x1]->[3x3]->[1x3]+[3x1]   [pool]->[1x1]
# 5A: 320,     384, 384, 384       448 384 384 384            192, 'AVE'
# 5A: 320,     384, 384, 384       448 384 384 384            192, 'AVE'
def addInceptionT3(model, name, bottom, p1_num_out, p2_num_out, p3_proj, p3_num_out, p4_num_out, pool):
    # p1:1x1
    model, top1 = addConvBnRelu(model, name='{}/p1_1x1'.format(name), bottom=bottom, num_output=p1_num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")

    # p2:[1x1]->([1x3]+[3x1])
    model, top2 = addConvBnRelu(model, name='{}/p2_1x1'.format(name), bottom=bottom, num_output=p2_num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top2a = addConvBnRelu(model, name='{}/p2_1x3'.format(name), bottom=top2, num_output=p2_num_out,
                                kernel_h=1, kernel_w=3, pad_h=0, pad_w=1, stride=1, filler="xavier")
    model, top2b = addConvBnRelu(model, name='{}/p2_3x1'.format(name), bottom=top2, num_output=p2_num_out,
                                kernel_h=3, kernel_w=1, pad_h=1, pad_w=0, stride=1, filler="xavier")
    p2_tops = [top2a, top2b]
    model,top2 = addConcat(model, name='{}/p2_concat'.format(name),bottoms=p2_tops)

    # p3:  1x1->3x3->([3x1] + [1x3])
    model, top3 = addConvBnRelu(model, name='{}/p3_1x1'.format(name), bottom=bottom, num_output=p3_proj,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    model, top3 = addConvBnRelu(model, name='{}/p3_3x3'.format(name), bottom=top3, num_output=p3_num_out,
                                kernel_size=3, stride=1, pad=1, filler="xavier")
    model, top3a = addConvBnRelu(model, name='{}/p3_1x3'.format(name), bottom=top3, num_output=p3_num_out,
                                kernel_h=1, kernel_w=3, pad_h=0, pad_w=1, stride=1, filler="xavier")
    model, top3b = addConvBnRelu(model, name='{}/p3_3x1'.format(name), bottom=top3, num_output=p3_num_out,
                                kernel_h=3, kernel_w=1, pad_h=1, pad_w=0, stride=1, filler="xavier")
    p3_tops = [top3a, top3b]
    model,top3 = addConcat(model, name='{}/p3_concat'.format(name),bottoms=p3_tops)

    # pool->[1x1]
    model, top4 = addPool(model, name='{}/p4_pool'.format(name), bottom=bottom,
                                kernel_size=3, stride=1, pad=1, pool_type=pool)
    model, top4 = addConvBnRelu(model, name='{}/p4_1x1'.format(name), bottom=top4, num_output=p4_num_out,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    # concat
    tops = [top1, top2, top3, top4]
    model, top = addConcat(model, name='{}/concat'.format(name), bottoms=tops)
    return model, top

#-----------------------------------------------------------------------------------

def addAuxLoss(model, name, bottom):

    model, top = addPool(model, name="{}/pool".format(name), bottom=bottom,
                                kernel_size=5, stride=3, pool_type="AVE")
    model, top = addConvBnRelu(model, name='{}/conv'.format(name), bottom=top, num_output=128,
                                kernel_size=1, stride=1, pad=0, filler="xavier")
    #    model, top = addDropoutl(model, name="dropout", bottom =top, ratio=0.5)
    model, top = addFC(model, name="{}/fc1".format(name), bottom=top, num_output=1024, filler='xavier')
    model, top = addActivation(model, name="{}/relu".format(name),bottom=top)
    model, top = addFC(model, name="{}/fc2".format(name), bottom=top, num_output=1000, filler='xavier')
    fc_top = top
    model, top = addSoftmaxLoss(model, name="{}/loss".format(name), bottom_1=fc_top, loss_weight=0.3)
    model, top = addAccuracy(model, name="{}/top-1".format(name),  bottom_1=fc_top, k=1)
    model, top = addAccuracy(model, name="{}/top-5".format(name), bottom_1=fc_top, k=5)

    return model, top

#------------------------------------------------------------------------------

def buildInception_v3():

    model = ""
    model = addHeader(model, name="Inception_v3")

    train_batch = 256
    test_batch  = 32
    crop_size = 299
    train_file="examples/imagenet/ilsvrc12_320x320_train_lmdb"
    test_file="examples/imagenet/ilsvrc12_320x320_val_lmdb"
    mean_file="data/ilsvrc12/imagenet_320x320_mean.binaryproto"
    model,data_top = addData(model, train_batch, test_batch, train_file, test_file, mean_file, crop_size)

    model,top = addConvBnRelu(model, 'conv1', data_top, num_output=32,
                              kernel_size=3, stride=2, pad=0, filler="xavier")
    # 149 x 149 x 32
    model,top = addConvBnRelu(model, 'conv2', top, num_output=32,
                              kernel_size=3, stride=1, pad=0, filler="xavier")
    # 147 x 147 x 32
    model,top = addConvBnRelu(model, 'conv3', top, num_output=64,
                              kernel_size=3, stride=1, pad=1, filler="xavier")
    # 147 x 147 x 64
    model,top = addPool(model, 'pool1', top, kernel_size=3, stride=2, pad=0, pool_type='MAX')
    # 73 x 73 x 64
    model,top = addConvBnRelu(model, 'conv4', top, num_output=80,
                              kernel_size=3, stride=1, pad=0, filler="xavier")
    # 73 x 73 x 80
    model,top = addConvBnRelu(model, 'conv5', top, num_output=192,
                              kernel_size=3, stride=2, pad=0, filler="xavier")
    # 71 x 71 x 192.
    model,top = addConvBnRelu(model, 'conv6', top, num_output=288,
                              kernel_size=3, stride=1, pad=1, filler="xavier")

    # 35 x 35 x 288:
    # Inception block T1
    #  64x[1x1] + 64x[1x1]->96x[3x3]->96x[3x3] + 48x[1x1]->64x[3x3]x[3x3] + [pool]->64x[1x1]
    #  64,        64->96->96,                   48->64,                     [AVE_3x3]->64
     # NOTE2: TF has instead of p2=[1x1]->[3x3] has p2= [1x1]->[3x3]->[3x3]

    model, top = addInceptionT1(model, "3A", top, 64, 64, 96, 48, 64, 64, 'AVE')
    model, top = addInceptionT1(model, "3B", top, 64, 64, 96, 48, 64, 64, 'AVE')
    model, top = addInceptionT1(model, "3C", top, 64, 64, 96, 48, 64, 64, 'AVE')

    # 17 x 17 x 768:
    # Reduction block 1 -------------------------------------------------------
    #  64x[1x1]->96x[3x3]->96x[3x3/2] + 384x[3x3/2] + [Max_3x3/2]
    model, top = addReductionR1(model, "3R", top, 64, 96, 96, 384, pool='MAX')

    # AuxLoss-1----------------------------------------------------------------
    model, aux1 = addAuxLoss(model, "loss1", top)

    # 17 x 17 x 768: Inception block T2 ---------------------------------------
    #  [1x1]  [1x1]->[1x7]->[7x1]   [1x1]->[1x7]->[7x1]->[1x7]->[7x1]  [pool]->[1x1]
    #  192,    128->128->192            128->128->128->128->192         'AVE'_>192
    #  192,    160->160->192            160->160->160->160->192         'AVE'_>192
    #  192,    160->160->192            160->160->160->160->192         'AVE'_>192
    #  192,    192->192->192            192->192->192->192->192         'AVE'_>192
    #  192,    192->192->192            192->192->192->192->192         'AVE'_>192

    model, top = addInceptionT2(model, '4A', top, 128, 192, 'AVE')
    model, top = addInceptionT2(model, '4B', top, 160, 192, 'AVE')
    model, top = addInceptionT2(model, '4C', top, 160, 192, 'AVE')
    model, top = addInceptionT2(model, '4D', top, 192, 192, 'AVE')
    model, top = addInceptionT2(model, '4E', top, 192, 192, 'AVE')

    # Reduction block 2 -------------------------------------------------------
    # 192x[1x1]->320x[3x3/2] + 192x[1x1]->192x[1x7]->192x[7x1]->192x[3x3/2] + [MAX_3x3/2]
    model, top = addReductionR2(model, "4R", top, 192, 320, pool='MAX')

    # AuxLoss-2----------------------------------------------------------------
    model, aux2 = addAuxLoss(model, "loss2", top)

    # 35 x 35 x 192:  Inception block T3---------------------------------------
    #  [1x1]   [1x1]->[1x3]+[3x1]   [1x1]->[3x3]->[1x3]+[3x1]   [pool]->[1x1]
    #  320,     384, 384, 384       448 384 384 384            192, 'AVE'
    model, top = addInceptionT3(model, '5A', top, 320, 384, 448, 384, 192, 'AVE')
    model, top = addInceptionT3(model, '5B', top, 320, 384, 448, 384, 192, 'AVE')

    # Loss3---------------------------------------------------------------------
    model, top = addPool(model, name="loss/pool", bottom=top, kernel_size=7, stride=1, pool_type="AVE")
#    model, top = addDropoutl(model, name="loss/dropout", bottom =top, ratio=0.5)
    model, top = addFC(model, name="loss/fc", bottom=top, num_output=1000, filler='xavier')
    fc_top = top
    model, top = addSoftmaxLoss(model, name="loss", bottom_1=fc_top)
    model, top = addAccuracy(model, name="accuracy/top-1", bottom_1=fc_top, k=1)
    model, top = addAccuracy(model, name="accuracy/top-5", bottom_1=fc_top, k=5)

    return model

#------------------------------------------------------------------------------

def main():

    model = buildInception_v3()
    fp = open("inception_v3.prototxt", 'w')
    fp.write(model)


if __name__ == '__main__':
    main()
