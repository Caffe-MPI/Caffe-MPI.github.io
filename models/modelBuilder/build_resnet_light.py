#!/usr/bin/env python
# ref: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"Deep Residual Learning for Image Recognition"
#      https://arxiv.org/pdf/1512.03385v1.pdf


import numpy

from layers import *

#------------------------------------------------------------------------------

def addRes_small(model, name , bottom, num_output, group, j, fix_dim, dilation = False):

    prefix="{name}.{j}.".format(name=name,j=str(j))
    block=""
    block, top = addConvBnRelu(model=block, name='{}conv1'.format(prefix), bottom=bottom, num_output=num_output,
                               kernel_size=3, group=group,
                               stride=2 if (fix_dim and (j==1)) else 1,
                               pad=j if dilation else 1,
                               dilation=j if dilation else 1)
    block, top = addConvBn(model=block, name='{}conv2'.format(prefix), bottom=top, num_output=num_output,
                               kernel_size=3, group=group, stride=1,
                               pad=j if dilation else 1,
                               dilation=j if dilation else 1)
    if (fix_dim and (j == 1)):
        block, res_top = addConvBn(model=block, name='{}skipConv'.format(prefix), bottom=bottom, num_output=num_output,
                               kernel_size=1, group=1, stride=2, pad=0)
    else:
        res_top = bottom
    block, top = addEltwise(model=block, name='{}sum'.format(prefix),
                            bottom_1=top, bottom_2=res_top, operation="SUM")
    block, top = addActivation(model=block, name="{}relu".format(prefix), bottom=top)

    model += block
    return model, top

#------------------------------------------------------------------------------------
# Resnet with less BN
#------------------------------------------------------------------------------------
def addRes_lightbn(model, name , bottom, num_output, group, j, fix_dim, dilation = False):

    prefix="{name}.{j}.".format(name=name,j=str(j))
    block=""
    block, top = addConvRelu(model=block, name='{}conv1'.format(prefix), bottom=bottom, num_output=num_output,
                               kernel_size=1, group=1, stride=2 if (fix_dim and (j==1)) else 1, pad=0)
    block, top = addConvRelu(model=block, name='{}conv2'.format(prefix), bottom=top, num_output=num_output,
                               kernel_size=3, group=group, stride=1, pad=1)
    block, top = addConv(model=block, name='{}conv3'.format(prefix), bottom=top, num_output=(num_output * 4),
                         bias_term=True,
                         kernel_size=1, group=1, stride=1, pad=0)

    if (j == 1):
        block, res_top = addConv(model=block, name='{}skipConv'.format(prefix), bottom=bottom,
                               num_output=(num_output * 4),
                               bias_term=True,  
                               kernel_size=1, group=1, stride=2 if fix_dim  else 1, pad=0)
    else:
        res_top = bottom

    block, top = addEltwise(model=block, name='{}sum'.format(prefix),
                            bottom_1=top, bottom_2=res_top, operation="SUM")
    if (j == 3):
        block, top = addBN(model=block, name='{}bn'.format(prefix), bottom=top)

    block, top = addActivation(model=block, name="{}relu".format(prefix), bottom=top)

    model += block
    return model, top

#------------------------------------------------------------------------------

def addResSuperBlock(model, bottom, i, num_subBlocks, num_output, group, fix_dim, net_type, dilation = False):
    name = "res{i}".format(i=str(i))
    model = addComment(model, comment=name)
    top=bottom
    for j in xrange(1, num_subBlocks + 1):
        if net_type=="small":
          model, top = addRes_small(model, name, bottom=top, num_output=num_output, group=group,
                                    j=j, fix_dim=fix_dim, dilation=dilation)
        else:
          model, top = addRes_lightbn(model, name, bottom=top, num_output=num_output, group=group,
                                    j=j, fix_dim=fix_dim, dilation=dilation)
    return model, top

#------------------------------------------------------------------------------

def print_netconfig(netConfig):

    header_str='''
# n:  ch\t:  s\t:  g\t: skip\t:'''

    num_blocks=netConfig.shape[0]
    for i in xrange(0, num_blocks):
        header_str += '''
# {i}:  {num_outputs}\t:  {subblocks}\t:  {group}\t:  {skip}\t:'''.format(
            i=str(i),
            num_outputs=str(netConfig[i,0]),
            subblocks  =str(netConfig[i,1]),
            group      =str(netConfig[i,2]),
            skip       =str(netConfig[i,3]) )

    header_str += "\n"
    return header_str

#------------------------------------------------------------------------------

def buildResidualModel(netConfig, name, net_type):

    model = ""
    model = addHeader(model, name=name)
    model += print_netconfig(netConfig)
    print(model)

    train_batch = 256 if net_type=="small" else 128
    test_batch  = 32
    model, last_top = addData(model, train_batch, test_batch)

    model, top = addConvRelu(model, name="conv1", bottom="data", num_output=64,
                               kernel_size=7, group=1, stride=2, pad=3)
    model, top = addPool(model, name="pool1", bottom=top,
                               kernel_size=3, stride=2, pool_type="MAX")

    num_blocks = len(netConfig)
    for i in xrange(1, num_blocks+1):
        num_output    = netConfig[i-1,0]
        num_subBlocks = netConfig[i-1,1]
        group = netConfig[i-1,2]
        fix_dim = (netConfig[i-1,3]==1)
        bottom=top
        model, top = addResSuperBlock(model, bottom, i+1, num_subBlocks, num_output, group, fix_dim, net_type)

    model, top = addPool(model, name="pool2", bottom=top, kernel_size=7, stride=1, pool_type="AVE")
#    model, top = addDropout(model, name="dropout", bottom=top, ratio=0.5)
    model, top = addFC(model, name="fc", bottom=top, num_output=1000, filler='msra')

    fc_top = top
    model, top = addSoftmaxLoss(model, name="loss", bottom_1=fc_top)
    model, top = addAccuracy(model, name="accuracy/top-1", bottom_1=fc_top, k=1)
    model, top = addAccuracy(model, name="accuracy/top-5", bottom_1=fc_top, k=5)

    return model

#------------------------------------------------------------------------------

def main():

    netConfig = numpy.matrix([
        [  64, 2, 1, 0],
        [ 128, 2, 1, 1],
        [ 256, 2, 1, 1],
        [ 512, 2, 1, 1]  ])

    model = buildResidualModel(netConfig, name="Resnet18", net_type="small")
    fp = open("resnet_18.prototxt", 'w')
    fp.write(model)

    netConfig = numpy.matrix([
        [ 64, 3, 1, 0],
        [128, 4, 1, 1],
        [256, 6, 1, 1],
        [512, 3, 1, 1]])
    model = buildResidualModel(netConfig, name="Resnet34", net_type="small")
    fp = open("resnet_34.prototxt", 'w')
    fp.write(model)

    netConfig = numpy.matrix([
        [ 64, 3, 1, 0],
        [128, 4, 1, 1],
        [256, 6, 1, 1],
        [512, 3, 1, 1]])
    model = buildResidualModel(netConfig, name="Resnet50", net_type="large")
    fp = open("resnet_50_lightbn.prototxt", 'w')
    fp.write(model)

if __name__ == '__main__':
    main()
