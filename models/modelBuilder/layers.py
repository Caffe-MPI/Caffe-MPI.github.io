#!/usr/bin/env python
#

#import numpy

#------------------------------------------------------------------------------

def  addHeader(model, name):

    header = '''name: "{name}"
'''.format(name=name)

    model += header
    return model

#------------------------------------------------------------------------------

def addData(model, train_batch=32, test_batch=32,
                 train_file="examples/imagenet/ilsvrc12_train_lmdb",
                 test_file = "examples/imagenet/ilsvrc12_val_lmdb",
                 mean_file = "data/ilsvrc12/imagenet_mean.binaryproto",
                 crop_size=224):

    layer = '''
layer {{
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {{
    source: "{train_file}"
    backend: LMDB
    batch_size: {train_batch}
  }}
  transform_param {{
    crop_size: {crop_size}
    mean_file: "{mean_file}"
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
    source: "{test_file}"
    backend: LMDB
    batch_size: {test_batch}
  }}
  transform_param {{
    mean_file: "{mean_file}"
    crop_size: {crop_size}
    mirror: false
  }}
  include: {{ phase: TEST }}
}}'''.format(train_batch=train_batch, test_batch=test_batch,
             train_file=train_file, mean_file=mean_file, test_file=test_file,
             crop_size=crop_size )

    model += layer
    return model, "data"

#------------------------------------------------------------------------------

def addConv(model, name, bottom, num_output,
                 kernel_size=0, kernel_h=0, kernel_w=0,
                 pad=0, pad_h=0, pad_w=0,
                 group=1, stride=1, dilation=1,
                 bias_term=False, filler="msra",
                 weight_sharing=False, weight_name="", bias_name="",
                 residual=False, residual_init = False):
    top=name
    layer = '''
layer {{
  name: "{name}"
  type: "Convolution"
  bottom: "{bottom}"
  top: "{top}"
  convolution_param {{
    num_output: {num_output}\n'''.format(name=name, bottom=bottom, top=top, num_output=num_output)

    if (kernel_size > 0):
        layer += '''    kernel_size: {}\n'''.format(kernel_size)
    if (kernel_h > 0):
        layer += '''    kernel_h: {}\n'''.format(kernel_h)
    if (kernel_w > 0):
        layer += '''    kernel_w: {}\n'''.format(kernel_w)
    if (pad > 0 ):
        layer += '''    pad: {}\n'''.format(pad)
    if (pad_h > 0 ):
        layer += '''    pad_h: {}\n'''.format(pad_h)
    if (pad_w > 0 ):
        layer += '''    pad_w: {}\n'''.format(pad_w)
    if (stride > 1):
        layer += '''    stride: {}\n'''.format(stride)

    if (dilation > 1):
        layer += '''    dilation: {}\n'''.format(dilation)
    if residual:
        layer += '''    residual: true\n'''
    if residual_init:
        layer += '''    residual_init: true\n'''

    layer += '''    weight_filler {{
      type: "{}"
    }}\n'''.format(filler)

#    layer = layer + '''
#    weight_filler {{
#      type: "{weight_filler}"
#    std: 0.010
#    }}'''.format(weight_filler=filler)

    if bias_term:
        layer += '''    bias_filler {
      type: "constant"
      value: 0
    }\n'''
    else:
        layer += '''    bias_term: false\n'''

    if (group>1):
        layer += '''    group: {}\n'''.format(group)

    layer += '''  }\n'''

    if weight_sharing:
        layer += '''  param {{ name: "{}" }}\n'''.format(weight_name)
        if bias_term:
            layer += '''  param {{ name: "{}" }}\n'''.format(bias_name)
    layer += '''}'''

    model += layer
    return model, top

#------------------------------------------------------------------------------

def addBN(model, name, bottom, top=None):

    top = name if top is None else top
    layer = '''
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
    model += layer
    return model, top

#------------------------------------------------------------------------------

def addActivation(model, name, bottom, type="ReLU"):
    layer = '''
layer {{
  name: "{name}"
  type: "{type}"
  bottom: "{bottom}"
  top: "{top}"
}}'''.format(name=name, top=bottom, bottom=bottom, type=type)
    model += layer
    return model, bottom

#------------------------------------------------------------------------------

def addConvRelu(model, name, bottom, num_output,
                kernel_size=0, kernel_h=0, kernel_w=0,
                pad=0, pad_h=0, pad_w=0,
                group=1, stride=1, dilation=1,
                filler="msra",
                weight_sharing=False, weight_name="", bias_name="",
                residual=False, residual_init=False):
    model, top = addConv(model=model, name=name, bottom=bottom, num_output=num_output,
                         kernel_size=kernel_size, kernel_h=kernel_h, kernel_w=kernel_w,
                         pad=pad, pad_h=pad_h, pad_w=pad_w,
                         group=group, stride=stride, dilation=dilation,
                         bias_term=True,filler=filler,
                         weight_sharing=weight_sharing, weight_name=weight_name, bias_name=bias_name,
                         residual=residual, residual_init=residual_init)
    model, top = addActivation(model=model, name="{}/relu".format(name), bottom=top, type="ReLU")
    return model, top

#------------------------------------------------------------------------------

def addConvBn(model, name, bottom, num_output,
              kernel_size=0, kernel_h=0, kernel_w=0,
              pad=0, pad_h=0, pad_w=0,
              group=1, stride=1, dilation=1,
              filler="msra", weight_sharing=False, weight_name="", bias_name="",
              residual=False, residual_init=False):

    model, top = addConv(model=model, name=name, bottom=bottom, num_output=num_output,
                         kernel_size=kernel_size, kernel_h=kernel_h, kernel_w=kernel_w,
                         pad=pad, pad_h=pad_h, pad_w=pad_w,
                         group=group, stride=stride, dilation=dilation,
                         bias_term=False,filler=filler,
                         weight_sharing=weight_sharing, weight_name=weight_name, bias_name=bias_name,
                         residual=residual, residual_init=residual_init)
    model, top = addBN(model, name="{}/bn".format(name), bottom=top)
    return model, top

#------------------------------------------------------------------------------

def addConvBnRelu(model, name, bottom, num_output,
                       kernel_size=0, kernel_h=0, kernel_w=0,
                       pad=0, pad_h=0, pad_w=0,
                       group=1, stride=1, dilation=1,
                       filler="msra", weight_sharing=False, weight_name="", bias_name="",
                       residual=False, residual_init=False):

    model, top = addConv(model=model, name=name, bottom=bottom, num_output=num_output,
                       kernel_size=kernel_size, kernel_h=kernel_h, kernel_w=kernel_w,
                       pad=pad, pad_h=pad_h, pad_w=pad_w,
                       group=group, stride=stride, dilation=dilation,
                       bias_term=False,filler=filler,
                       weight_sharing=weight_sharing, weight_name=weight_name, bias_name=bias_name,
                       residual=residual, residual_init=residual_init)
    model, top = addBN(model, name="{}/bn".format(name), bottom=top)
    model, top = addActivation(model=model, name="{}/relu".format(name), bottom=top, type="ReLU")
    return model, top

#------------------------------------------------------------------------------

def addBnRelu(model, name, bottom):

    model, top = addBN(model, name="{}/bn".format(name), bottom=bottom, top="{}bn".format(name))
    model, top = addActivation(model, name="{}/relu".format(name), bottom=top, type="ReLU")
    return model, top

#------------------------------------------------------------------------------

def addPool(model, name, bottom, kernel_size, stride, pool_type, pad=0):

    layer = '''
layer {{
  name: "{name}"
  type: "Pooling"
  bottom: "{bottom}"
  top: "{top}"
  pooling_param {{
    pool: {pool_type}
    kernel_size: {kernel_size}\n'''.format(name=name, top=name, bottom=bottom,
                               pool_type=pool_type, kernel_size=kernel_size)

    if (stride>1):
        layer += '''    stride: {}\n'''.format(stride)

    if (pad>0):
        layer += '''    pad: {}\n'''.format(pad)

    layer+='''  }\n}'''
    model += layer
    return model, name

#------------------------------------------------------------------------------
def addSlice(model, name, bottom, num_output, group, slice_offset):

    layer = '''
layer {{
  name: "{name}"
  type: "Slice""
  bottom: "{bottom}"
'''.format(name=name, bottom=bottom)

    tops = []
    for i in xrange(1, group + 1):
        top="name_{}".format(i)
        layer +=  '''  top: {}\n'''.format(top)
        tops.append(top)

 #   print tops

    layer += '''  slice_param {
    axis: 1 '''

    slice_point = slice_offset;
    for i in xrange(1, group ):
        layer += '''    slice_point: {}\n'''. format(slice_point)
        slice_point += slice_offset

    layer += '''  }
}'''

    model += layer
    return model, tops

#------------------------------------------------------------------------------

def addFC(model, name, bottom, num_output, filler="gaussian"):

    layer = '''
layer {{
  name: "{name}"
  type: "InnerProduct"
  bottom: "{bottom}"
  top: "{top}"
  inner_product_param {{
    num_output: {num_output}
    weight_filler {{
      type: "{weight_filler_type}"
'''.format(name=name, top=name, bottom=bottom, num_output=num_output, weight_filler_type=filler)

    if (filler == "gaussian"):
        layer +='''      std: 0.01\n'''

    layer +='''    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}'''

    model += layer
    return model, name

#------------------------------------------------------------------------------

def addEltwise(model, name, bottom_1, bottom_2, operation="SUM"):

    layer = '''
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

    model += layer
    return model, name

#------------------------------------------------------------------------------

def addMultiEltwise(model, name, bottoms, operation="SUM"):

#    num_columns=len(bottoms)
    top = name
    model += '''
layer {{
  name: "{name}"
  type: "Eltwise"
  top: "{top}" '''.format(name=name, top=top)

    for bottom in bottoms:
        model += '''  bottom: "{}" \n'''.format(bottom)

    model += '''  eltwise_param {{ operation: {} }}\n}}'''.format(operation)

    return model, top

#------------------------------------------------------------------------------

def addSplit(model, name, bottom, num_tops):

    layer = '''
layer {{
  name: "{name}"
  type: "Split"
  bottom: "{bottom}"
'''.format(name=name, bottom=bottom)

    tops =[]
    for i in xrange(1, num_tops+1 ):
        top="{name}.{i}".format(name=name, i=i)
        tops.append(top)
        layer += '''  top: "{}"\n'''.format(top)

    layer += '''}'''
    model +=layer
    return model, tops

#------------------------------------------------------------------------------
def addConcat(model, name, bottoms):

    top = name
    model += '''
layer {{
  name: "{name}"
  type: "Concat"\n'''.format(name=name)

    for bottom in bottoms:
        model += '''  bottom: "{}"\n'''.format(bottom)

    model += '''  top: "{top}"
}}'''. format(top=top)

    return model, top
#------------------------------------------------------------------------------

def addReplicator(model, name, bottom, multiplier):
    model, tops = addSplit(model, "{}.r".format(name), bottom, multiplier)
    model, top  = addConcat(model, "{}.c".format(name), tops)

    return model, top
#------------------------------------------------------------------------------

def addDropout(train_val, name, bottom, ratio=0.5):

    layer_str = '''
layer {{
  name: "{name}"
  type: "Dropout"
  bottom: "{bottom}"
  top: "{top}"
  dropout_param {{
    dropout_ratio: {ratio}
  }}
}}'''.format(name=name, bottom=bottom, top=name, ratio=ratio)

    train_val += layer_str
    return train_val, name

#------------------------------------------------------------------------------

def addSoftmaxLoss(model, name, bottom_1, bottom_2="label", loss_weight=1.0):

    layer = '''
layer {{
  name: "{name}"
  type: "SoftmaxWithLoss"
  bottom: "{bottom_1}"
  bottom: "{bottom_2}"
  top: "{top}"\n'''.format(name=name, top=name, bottom_1=bottom_1, bottom_2=bottom_2)
    if (loss_weight!=1.0):
        layer += '''  loss_weight:{}\n'''.format(loss_weight)
    layer += ''' }\n'''

    model += layer
    return model, name

#------------------------------------------------------------------------------

def addAccuracy(model, name, bottom_1, bottom_2="label", k=1):

    layer='''
layer {{
  name: "{name}"
  type: "Accuracy"
  bottom: "{bottom_1}"
  bottom: "{bottom_2}"
  top: "{top}"
  accuracy_param {{ top_k: {k} }}
#  include {{ phase: TEST }}
}}'''.format(name=name, top=name, bottom_1=bottom_1, bottom_2=bottom_2, k=k)
    model += layer
    return model, name

#------------------------------------------------------------------------------

def addComment(model, comment):
    model += "\n#\n# {comment}\n#".format(comment=comment)
    return model
