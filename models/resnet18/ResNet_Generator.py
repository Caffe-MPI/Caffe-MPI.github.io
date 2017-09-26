#!/usr/bin/env python
# ref: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"Deep Residual Learning for Image Recognition"
#     https://arxiv.org/pdf/1512.03385v1.pdf
#
ResNetConfig={
   "16":[ "16", "small", 1, 1,  1, 1],
   "18":[ "18", "small", 2, 2,  2, 2],
   "34":[ "34", "small", 3, 4,  6, 3],
   "50":[ "50", "large", 3, 4,  6, 3],
  "101":["101", "large", 3, 4, 23, 3],
  "152":["152", "large", 3, 8, 36, 3]
}


def genDataLayer(train_val, number):
    layer_str = '''name: "Resnet{number}"
layer {{
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {{
    source: "examples/imagenet/ilsvrc12_train_lmdb"
    backend: LMDB
    batch_size: 32
  }}
  transform_param {{
    crop_size: 224
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
    mirror: true
  }}
  include: {{ phase: TRAIN }}
}}
layer {{
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {{
    source: "examples/imagenet/ilsvrc12_val_lmdb"
    backend: LMDB
    batch_size: 32
  }}
  transform_param {{
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
    crop_size: 224
    mirror: false
  }}
  include: {{ phase: TEST }}
}}'''.format(number=number)
    train_val += layer_str
    return train_val, "data"


def genConvLayer(train_val, name, bottom, kernel_size, num_output, stride, pad, bias_term=False,filler="msra"):

    layer_str = '''
layer {{
  name: "{name}"
  type: "Convolution"
  bottom: "{bottom}"
  top: "{top}"
  convolution_param {{
    num_output: {num_output}
    kernel_size: {kernel_size}
    stride: {stride}
    pad: {pad}
    weight_filler {{
      type: "{weight_filler_type}"
      std: 0.010
    }}'''.format(name=name, top=name, bottom=bottom, kernel_size=kernel_size,
             num_output=num_output, pad=pad, stride=stride, weight_filler_type=filler)

    if bias_term:
        layer_str = layer_str + \
'''
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}'''
    else :
        layer_str = layer_str + \
'''
    bias_term: false
  }
}'''
    train_val += layer_str
    return train_val, name


def genBNLayer(train_val, name, bottom, top=None):
    top = name if top is None else top
    layer_str = '''
layer {{
  name: "{name}"
  type: "BatchNorm"
  bottom: "{bottom}"
  top: "{top}"
  batch_norm_param {{
    moving_average_fraction: 0.9
    eps: 0.0001
    scale_bias: true
  }}
}}'''.format(name=name, top=top, bottom=bottom)
    train_val += layer_str
    return train_val, top

# def genScaleLayer(train_val, name, bottom):
#     layer_str = '''
# layer {{
#   name: "{name}"
#   type: "Scale"
#   top: "{top}"
#   bottom: "{bottom}"
#   scale_param {{
#     bias_term: true # TODO
#   }}
# }}'''.format(name=name, top=bottom, bottom=bottom)

#     train_val += layer_str
#     return train_val, bottom


def genActivationLayer(train_val, name, bottom, type="ReLU"):
    layer_str = '''
layer {{
  name: "{name}"
  type: "{type}"
  bottom: "{bottom}"
  top: "{top}"
}}'''.format(name=name, top=bottom, bottom=bottom, type=type)
    train_val += layer_str
    return train_val, bottom


def genConvBnLayer(train_val, name, bottom, kernel_size, num_output, stride, pad, filler="msra"):
    train_val, last_top = genConvLayer(train_val=train_val, name=name, bottom=bottom,
        kernel_size=kernel_size, num_output=num_output, stride=stride, pad=pad, bias_term=False,filler=filler)
    train_val, last_top = genBNLayer(train_val=train_val, name="{name}_bn".format(name=name), bottom=last_top)
    return train_val, last_top


def genConvBnReluLayer(train_val, name, bottom, kernel_size, num_output, stride, pad, filler="msra", activation_type="ReLU"):
    train_val, last_top = genConvBnLayer(train_val=train_val, name=name, bottom=bottom,
        kernel_size=kernel_size, num_output=num_output, stride=stride, pad=pad, filler=filler)
    train_val, last_top = genActivationLayer(train_val=train_val, name="{name}_relu".format(name=name), bottom=last_top, type=activation_type)
    return train_val, last_top


def genBnReluLayer(train_val, name, bottom, activation_type="ReLU"):
    train_val, last_top = genBNLayer(train_val=train_val, name="{name}bn".format(name=name), bottom=bottom, top="{name}bn".format(name=name))
    train_val, last_top = genActivationLayer(train_val=train_val, name="{name}relu".format(name=name), bottom=last_top, type=activation_type)
    return train_val, last_top


def genPoolLayer(train_val, name, bottom, kernel_size, stride, pool_type):
    layer_str = '''
layer {{
  name: "{name}"
  type: "Pooling"
  bottom: "{bottom}"
  top: "{top}"
  pooling_param {{
    pool: {pool_type}
    kernel_size: {kernel_size}
    stride: {stride}
  }}
}}'''.format(name=name, top=name, bottom=bottom, pool_type=pool_type, kernel_size=kernel_size, stride=stride)
    train_val += layer_str
    return train_val, name


def genFCLayer(train_val, name, bottom, num_output, filler="gaussian"):
    layer_str = '''
layer {{
  name: "{name}"
  type: "InnerProduct"
  bottom: "{bottom}"
  top: "{top}"
  inner_product_param {{
    num_output: {num_output}
    weight_filler {{
      type: "{weight_filler_type}"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}'''.format(name=name, top=name, bottom=bottom, num_output=num_output, weight_filler_type=filler)
    train_val += layer_str
    return train_val, name


def genEltwiseLayer(train_val, name, bottom_1, bottom_2, operation="SUM"):
    layer_str = '''
layer {{
  name: "{name}"
  type: "Eltwise"
  bottom: "{bottom_1}"
  bottom: "{bottom_2}"
  top: "{top}"
  eltwise_param {{
    operation: {operation}
  }}
}}'''.format(name=name, top=name, bottom_1=bottom_1, bottom_2=bottom_2, operation=operation)
    train_val += layer_str
    return train_val, name


def genSoftmaxLossLayer(train_val, name, bottom_1, bottom_2="label"):
    layer_str = '''
layer {{
  name: "{name}"
  type: "SoftmaxWithLoss"
  bottom: "{bottom_1}"
  bottom: "{bottom_2}"
  top: "{top}"
}}'''.format(name=name, top=name, bottom_1=bottom_1, bottom_2=bottom_2)
    train_val += layer_str
    return train_val, name


def genAccuracyLayer(train_val, name, bottom_1, bottom_2="label", k=1):
    layer_str='''
layer {{
  type: "Accuracy"
  name: "{name}"
  bottom: "{bottom_1}"
  bottom: "{bottom_2}"
  top: "{top}"
  accuracy_param {{
    top_k: {k}
  }}
  include {{ phase: TEST }}
}}'''.format(name=name, top=name, bottom_1=bottom_1, bottom_2=bottom_2, k=k)
    train_val += layer_str
    return train_val, name


def addComment(train_val, comment):
    train_val += "\n#\n# {comment}\n#".format(comment=comment)
    return train_val


def digit_to_char(digit):
    return chr(ord('A') + digit)


def str_base(number, base):
    if number < 0:
        return '-' + str_base(-number, base)
    (d, m) = divmod(number, base)
    if d > 0:
        return str_base(d, base) + digit_to_char(m)
    return (digit_to_char(m).lower())


def genRes2(train_val, last_top, small, i, fix_dim):
#    prefix="res2.{i}_".format(i=str_base(i-1, 26))
    prefix = "res2.{i}.".format(i=str(i))
    branch_str=""
    res_last_top=last_top
    branch_last_top=last_top

    if small:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=64, stride=1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=64, stride=1, pad=1)
    else:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=64, stride=1, pad=0)
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=64, stride=1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv3'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=256, stride=1, pad=0)

    if small==False:
       branch_str, res_last_top = genConvBnLayer(train_val=branch_str, name='{}skipConv'.format(prefix), bottom=res_last_top,
         kernel_size=1, num_output=64 if small else 256, stride=1, pad=0)

    branch_str, last_top = genEltwiseLayer(train_val=branch_str, name='{}eltwise'.format(prefix),
      bottom_1=branch_last_top, bottom_2=res_last_top, operation="SUM")
    branch_str, last_top = genActivationLayer(train_val=branch_str, name="{}relu".format(prefix), bottom=last_top)

    train_val += branch_str
    return train_val, last_top


def genRes3(train_val, last_top, small, i, fix_dim):
#    prefix="res3{i}_".format(i=str_base(i-1, 26))
    prefix="res3.{i}.".format(i=str(i))
    branch_str=""
    res_last_top=last_top
    branch_last_top=last_top

    if small:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=128, stride=2 if i==1 else 1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=128, stride=1, pad=1)
    else:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=128, stride=2 if i==1 else 1, pad=0)
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=128, stride=1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv3'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=512, stride=1, pad=0)

    if fix_dim:
        branch_str, res_last_top = genConvBnLayer(train_val=branch_str, name='{}skipConv'.format(prefix), bottom=res_last_top,
          kernel_size=1, num_output=128 if small else 512, stride=2, pad=0)

    branch_str, last_top = genEltwiseLayer(train_val=branch_str, name='{}eltwise'.format(prefix),
      bottom_1=branch_last_top, bottom_2=res_last_top, operation="SUM")
    branch_str, last_top = genActivationLayer(train_val=branch_str, name="{}relu".format(prefix), bottom=last_top)

    train_val += branch_str
    return train_val, last_top


def genRes4(train_val, last_top, small, i, fix_dim):
#    prefix="res4{i}_".format(i=str_base(i-1, 26))
    prefix="res4.{i}.".format(i=str(i))
    branch_str=""
    res_last_top=last_top
    branch_last_top=last_top

    if small:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=256, stride=2 if i==1 else 1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=256, stride=1, pad=1)
    else:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=256, stride=2 if i==1 else 1, pad=0)
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=256, stride=1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv3'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=1024, stride=1, pad=0)

    if fix_dim:
        branch_str, res_last_top = genConvBnLayer(train_val=branch_str, name='{}skipConv'.format(prefix), bottom=res_last_top,
          kernel_size=1, num_output=256 if small else 1024, stride=2, pad=0)

    branch_str, last_top = genEltwiseLayer(train_val=branch_str, name='{}eltwise'.format(prefix),
      bottom_1=branch_last_top, bottom_2=res_last_top, operation="SUM")
    branch_str, last_top = genActivationLayer(train_val=branch_str, name="{}relu".format(prefix), bottom=last_top)

    train_val += branch_str
    return train_val, last_top


def genRes5(train_val, last_top, small, i, fix_dim):
#    prefix="res5{i}_".format(i=str_base(i-1, 26))
    prefix="res5.{i}.".format(i=str(i))
    branch_str=""
    res_last_top=last_top
    branch_last_top=last_top

    if small:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=512, stride=2 if i==1 else 1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=512, stride=1, pad=1)
    else:
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv1'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=512, stride=2 if i==1 else 1, pad=0)
        branch_str, branch_last_top = genConvBnReluLayer(train_val=branch_str, name='{}conv2'.format(prefix), bottom=branch_last_top,
          kernel_size=3, num_output=512, stride=1, pad=1)
        branch_str, branch_last_top = genConvBnLayer(train_val=branch_str, name='{}conv3'.format(prefix), bottom=branch_last_top,
          kernel_size=1, num_output=2048, stride=1, pad=0)

    if fix_dim:
        branch_str, res_last_top = genConvBnLayer(train_val=branch_str, name='{}skipConv'.format(prefix), bottom=res_last_top,
          kernel_size=1, num_output=512 if small else 2048, stride=2, pad=0)

    branch_str, last_top = genEltwiseLayer(train_val=branch_str, name='{}eltwise'.format(prefix),
      bottom_1=branch_last_top, bottom_2=res_last_top, operation="SUM")
    branch_str, last_top = genActivationLayer(train_val=branch_str, name="{}relu".format(prefix), bottom=last_top)

    train_val += branch_str
    return train_val, last_top


def genTrainVal(network):
    train_val = ""
    train_val, last_top = genDataLayer(train_val=train_val, number=network[0])

    train_val = addComment(train_val=train_val, comment="Res1")
    train_val, last_top = genConvBnReluLayer(train_val=train_val, name="conv1", bottom="data", kernel_size=7, num_output=64, stride=2, pad=3)
    train_val, last_top = genPoolLayer(train_val=train_val, name="pool1", bottom=last_top, kernel_size=3, stride=2, pool_type="MAX")

    train_val = addComment(train_val=train_val, comment="ResBlock2")
    for i in xrange(1, network[2]+1):
        train_val, last_top = genRes2(train_val=train_val, last_top=last_top, small=network[1] is "small", i=i, fix_dim=False)

    train_val = addComment(train_val=train_val, comment="ResBlock3")
    for i in xrange(1, network[3]+1):
        train_val, last_top = genRes3(train_val=train_val, last_top=last_top, small=network[1] is "small", i=i, fix_dim=i==1)

    train_val = addComment(train_val=train_val, comment="ResBlock4")
    for i in xrange(1, network[4]+1):
        train_val, last_top = genRes4(train_val=train_val, last_top=last_top, small=network[1] is "small", i=i, fix_dim=i==1)

    train_val = addComment(train_val=train_val, comment="ResBlock5")
    for i in xrange(1, network[5]+1):
        train_val, last_top = genRes5(train_val=train_val, last_top=last_top, small=network[1] is "small", i=i, fix_dim=i==1)

    train_val, last_top = genPoolLayer(train_val=train_val, name="pool2", bottom=last_top, kernel_size=7, stride=1, pool_type="AVE")
    train_val, last_top = genFCLayer  (train_val=train_val, name="fc", bottom=last_top, num_output=1000, filler='msra')

    fc_top = last_top
    train_val, last_top = genSoftmaxLossLayer(train_val=train_val, name="loss", bottom_1=fc_top)
    train_val, last_top = genAccuracyLayer(train_val=train_val, name="accuracy/top-1", bottom_1=fc_top, k=1)
    train_val, last_top = genAccuracyLayer(train_val=train_val, name="accuracy/top-5", bottom_1=fc_top, k=5)
    return train_val


def main():
    for net in ResNetConfig.keys():
        network_str = genTrainVal(ResNetConfig[net])
#       with open("./models/train_val_{}.prototxt".format(net), 'w') as fp:
#            fp.write(network_str)
        fp = open("./models/train_val_{}.prototxt".format(net), 'w')
        fp.write(network_str)


if __name__ == '__main__':
    main()
