node {
  name: "ae_input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 64
        }
        dim {
          size: 64
        }
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\007\000\000\000\020\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.17025130987167358
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.17025130987167358
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d/kernel/Initializer/random_uniform/max"
  input: "conv2d/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
}
node {
  name: "conv2d/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d/kernel/Initializer/random_uniform/mul"
  input: "conv2d/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
}
node {
  name: "conv2d/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d/kernel"
    }
  }
}
node {
  name: "conv2d/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d/kernel"
}
node {
  name: "conv2d/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d/kernel"
  input: "conv2d/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d/bias"
    }
  }
}
node {
  name: "conv2d/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d/bias"
}
node {
  name: "conv2d/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d/bias"
  input: "conv2d/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d/Conv2D/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/Conv2D"
  op: "Conv2D"
  input: "ae_input"
  input: "conv2d/Conv2D/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d/BiasAdd"
  op: "BiasAdd"
  input: "conv2d/Conv2D"
  input: "conv2d/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d/Relu"
  op: "Relu"
  input: "conv2d/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1178511306643486
      }
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1178511306643486
      }
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_1/kernel/Initializer/random_uniform/max"
  input: "conv2d_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_1/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_1/kernel/Initializer/random_uniform/mul"
  input: "conv2d_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_1/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_1/kernel"
    }
  }
}
node {
  name: "conv2d_1/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_1/kernel"
}
node {
  name: "conv2d_1/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_1/kernel"
  input: "conv2d_1/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_1/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_1/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_1/bias"
    }
  }
}
node {
  name: "conv2d_1/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_1/bias"
}
node {
  name: "conv2d_1/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_1/bias"
  input: "conv2d_1/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_1/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/filter_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/required_space_to_batch_paddings/input_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/required_space_to_batch_paddings/paddings"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/required_space_to_batch_paddings/crops"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/SpaceToBatchND/block_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/SpaceToBatchND/paddings"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/SpaceToBatchND"
  op: "SpaceToBatchND"
  input: "conv2d/Relu"
  input: "conv2d_1/SpaceToBatchND/block_shape"
  input: "conv2d_1/SpaceToBatchND/paddings"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_1/Conv2D/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_1/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/Conv2D"
  op: "Conv2D"
  input: "conv2d_1/SpaceToBatchND"
  input: "conv2d_1/Conv2D/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_1/BatchToSpaceND/block_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\002\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/BatchToSpaceND/crops"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_1/BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "conv2d_1/Conv2D"
  input: "conv2d_1/BatchToSpaceND/block_shape"
  input: "conv2d_1/BatchToSpaceND/crops"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tcrops"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_1/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_1/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_1/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_1/BatchToSpaceND"
  input: "conv2d_1/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_1/Relu"
  op: "Relu"
  input: "conv2d_1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0833333358168602
      }
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0833333358168602
      }
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_2/kernel/Initializer/random_uniform/max"
  input: "conv2d_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_2/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_2/kernel/Initializer/random_uniform/mul"
  input: "conv2d_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_2/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_2/kernel"
    }
  }
}
node {
  name: "conv2d_2/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_2/kernel"
}
node {
  name: "conv2d_2/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_2/kernel"
  input: "conv2d_2/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_2/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_2/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_2/bias"
    }
  }
}
node {
  name: "conv2d_2/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_2/bias"
}
node {
  name: "conv2d_2/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_2/bias"
  input: "conv2d_2/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_2/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/filter_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/required_space_to_batch_paddings/input_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/required_space_to_batch_paddings/paddings"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/required_space_to_batch_paddings/crops"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/SpaceToBatchND/block_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/SpaceToBatchND/paddings"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/SpaceToBatchND"
  op: "SpaceToBatchND"
  input: "conv2d_1/Relu"
  input: "conv2d_2/SpaceToBatchND/block_shape"
  input: "conv2d_2/SpaceToBatchND/paddings"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_2/Conv2D/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_2/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/Conv2D"
  op: "Conv2D"
  input: "conv2d_2/SpaceToBatchND"
  input: "conv2d_2/Conv2D/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_2/BatchToSpaceND/block_shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\004\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/BatchToSpaceND/crops"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_2/BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "conv2d_2/Conv2D"
  input: "conv2d_2/BatchToSpaceND/block_shape"
  input: "conv2d_2/BatchToSpaceND/crops"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tcrops"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_2/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_2/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_2/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_2/BatchToSpaceND"
  input: "conv2d_2/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_2/Relu"
  op: "Relu"
  input: "conv2d_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0589255653321743
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0589255653321743
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/max"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/mul"
  input: "conv2d_transpose/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose/kernel"
    }
  }
}
node {
  name: "conv2d_transpose/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose/kernel"
}
node {
  name: "conv2d_transpose/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose/kernel"
  input: "conv2d_transpose/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_transpose/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose/bias"
    }
  }
}
node {
  name: "conv2d_transpose/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose/bias"
}
node {
  name: "conv2d_transpose/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose/bias"
  input: "conv2d_transpose/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/Shape"
  op: "Shape"
  input: "conv2d_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice"
  op: "StridedSlice"
  input: "conv2d_transpose/Shape"
  input: "conv2d_transpose/strided_slice/stack"
  input: "conv2d_transpose/strided_slice/stack_1"
  input: "conv2d_transpose/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_1"
  op: "StridedSlice"
  input: "conv2d_transpose/Shape"
  input: "conv2d_transpose/strided_slice_1/stack"
  input: "conv2d_transpose/strided_slice_1/stack_1"
  input: "conv2d_transpose/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_2/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_2/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/strided_slice_2"
  op: "StridedSlice"
  input: "conv2d_transpose/Shape"
  input: "conv2d_transpose/strided_slice_2/stack"
  input: "conv2d_transpose/strided_slice_2/stack_1"
  input: "conv2d_transpose/strided_slice_2/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/mul"
  op: "Mul"
  input: "conv2d_transpose/strided_slice_1"
  input: "conv2d_transpose/mul/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose/mul_1"
  op: "Mul"
  input: "conv2d_transpose/strided_slice_2"
  input: "conv2d_transpose/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose/stack/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 128
      }
    }
  }
}
node {
  name: "conv2d_transpose/stack"
  op: "Pack"
  input: "conv2d_transpose/strided_slice"
  input: "conv2d_transpose/mul"
  input: "conv2d_transpose/mul_1"
  input: "conv2d_transpose/stack/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose/conv2d_transpose/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/conv2d_transpose"
  op: "Conv2DBackpropInput"
  input: "conv2d_transpose/stack"
  input: "conv2d_transpose/conv2d_transpose/ReadVariableOp"
  input: "conv2d_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_transpose/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_transpose/conv2d_transpose"
  input: "conv2d_transpose/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_transpose/Relu"
  op: "Relu"
  input: "conv2d_transpose/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0589255653321743
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0589255653321743
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/max"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/mul"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_1/kernel"
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_1/kernel"
}
node {
  name: "conv2d_transpose_1/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "conv2d_transpose_1/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_1/bias"
    }
  }
}
node {
  name: "conv2d_transpose_1/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_1/bias"
}
node {
  name: "conv2d_transpose_1/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "conv2d_transpose_1/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/Shape"
  op: "Shape"
  input: "conv2d_transpose/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice"
  op: "StridedSlice"
  input: "conv2d_transpose_1/Shape"
  input: "conv2d_transpose_1/strided_slice/stack"
  input: "conv2d_transpose_1/strided_slice/stack_1"
  input: "conv2d_transpose_1/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_1"
  op: "StridedSlice"
  input: "conv2d_transpose_1/Shape"
  input: "conv2d_transpose_1/strided_slice_1/stack"
  input: "conv2d_transpose_1/strided_slice_1/stack_1"
  input: "conv2d_transpose_1/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_2/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_2/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/strided_slice_2"
  op: "StridedSlice"
  input: "conv2d_transpose_1/Shape"
  input: "conv2d_transpose_1/strided_slice_2/stack"
  input: "conv2d_transpose_1/strided_slice_2/stack_1"
  input: "conv2d_transpose_1/strided_slice_2/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_1/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/mul"
  op: "Mul"
  input: "conv2d_transpose_1/strided_slice_1"
  input: "conv2d_transpose_1/mul/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_1/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/mul_1"
  op: "Mul"
  input: "conv2d_transpose_1/strided_slice_2"
  input: "conv2d_transpose_1/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_1/stack/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 64
      }
    }
  }
}
node {
  name: "conv2d_transpose_1/stack"
  op: "Pack"
  input: "conv2d_transpose_1/strided_slice"
  input: "conv2d_transpose_1/mul"
  input: "conv2d_transpose_1/mul_1"
  input: "conv2d_transpose_1/stack/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_1/conv2d_transpose/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/conv2d_transpose"
  op: "Conv2DBackpropInput"
  input: "conv2d_transpose_1/stack"
  input: "conv2d_transpose_1/conv2d_transpose/ReadVariableOp"
  input: "conv2d_transpose/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_transpose_1/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_1/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_transpose_1/conv2d_transpose"
  input: "conv2d_transpose_1/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_transpose_1/Relu"
  op: "Relu"
  input: "conv2d_transpose_1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0833333358168602
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0833333358168602
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/max"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/mul"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_2/kernel"
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_2/kernel"
}
node {
  name: "conv2d_transpose_2/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "conv2d_transpose_2/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_2/bias"
    }
  }
}
node {
  name: "conv2d_transpose_2/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_2/bias"
}
node {
  name: "conv2d_transpose_2/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "conv2d_transpose_2/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/Shape"
  op: "Shape"
  input: "conv2d_transpose_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice"
  op: "StridedSlice"
  input: "conv2d_transpose_2/Shape"
  input: "conv2d_transpose_2/strided_slice/stack"
  input: "conv2d_transpose_2/strided_slice/stack_1"
  input: "conv2d_transpose_2/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_1"
  op: "StridedSlice"
  input: "conv2d_transpose_2/Shape"
  input: "conv2d_transpose_2/strided_slice_1/stack"
  input: "conv2d_transpose_2/strided_slice_1/stack_1"
  input: "conv2d_transpose_2/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_2/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_2/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/strided_slice_2"
  op: "StridedSlice"
  input: "conv2d_transpose_2/Shape"
  input: "conv2d_transpose_2/strided_slice_2/stack"
  input: "conv2d_transpose_2/strided_slice_2/stack_1"
  input: "conv2d_transpose_2/strided_slice_2/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_2/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/mul"
  op: "Mul"
  input: "conv2d_transpose_2/strided_slice_1"
  input: "conv2d_transpose_2/mul/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_2/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/mul_1"
  op: "Mul"
  input: "conv2d_transpose_2/strided_slice_2"
  input: "conv2d_transpose_2/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_2/stack/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 32
      }
    }
  }
}
node {
  name: "conv2d_transpose_2/stack"
  op: "Pack"
  input: "conv2d_transpose_2/strided_slice"
  input: "conv2d_transpose_2/mul"
  input: "conv2d_transpose_2/mul_1"
  input: "conv2d_transpose_2/stack/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_2/conv2d_transpose/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/conv2d_transpose"
  op: "Conv2DBackpropInput"
  input: "conv2d_transpose_2/stack"
  input: "conv2d_transpose_2/conv2d_transpose/ReadVariableOp"
  input: "conv2d_transpose_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_transpose_2/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_2/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_transpose_2/conv2d_transpose"
  input: "conv2d_transpose_2/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_transpose_2/Relu"
  op: "Relu"
  input: "conv2d_transpose_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1178511306643486
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1178511306643486
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/max"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/mul"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_3/kernel"
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_3/kernel"
}
node {
  name: "conv2d_transpose_3/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "conv2d_transpose_3/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_transpose_3/bias"
    }
  }
}
node {
  name: "conv2d_transpose_3/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_3/bias"
}
node {
  name: "conv2d_transpose_3/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "conv2d_transpose_3/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/Shape"
  op: "Shape"
  input: "conv2d_transpose_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice"
  op: "StridedSlice"
  input: "conv2d_transpose_3/Shape"
  input: "conv2d_transpose_3/strided_slice/stack"
  input: "conv2d_transpose_3/strided_slice/stack_1"
  input: "conv2d_transpose_3/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_1"
  op: "StridedSlice"
  input: "conv2d_transpose_3/Shape"
  input: "conv2d_transpose_3/strided_slice_1/stack"
  input: "conv2d_transpose_3/strided_slice_1/stack_1"
  input: "conv2d_transpose_3/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_2/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_2/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_2/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/strided_slice_2"
  op: "StridedSlice"
  input: "conv2d_transpose_3/Shape"
  input: "conv2d_transpose_3/strided_slice_2/stack"
  input: "conv2d_transpose_3/strided_slice_2/stack_1"
  input: "conv2d_transpose_3/strided_slice_2/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "conv2d_transpose_3/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/mul"
  op: "Mul"
  input: "conv2d_transpose_3/strided_slice_1"
  input: "conv2d_transpose_3/mul/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_3/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/mul_1"
  op: "Mul"
  input: "conv2d_transpose_3/strided_slice_2"
  input: "conv2d_transpose_3/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv2d_transpose_3/stack/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 16
      }
    }
  }
}
node {
  name: "conv2d_transpose_3/stack"
  op: "Pack"
  input: "conv2d_transpose_3/strided_slice"
  input: "conv2d_transpose_3/mul"
  input: "conv2d_transpose_3/mul_1"
  input: "conv2d_transpose_3/stack/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_transpose_3/conv2d_transpose/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/conv2d_transpose"
  op: "Conv2DBackpropInput"
  input: "conv2d_transpose_3/stack"
  input: "conv2d_transpose_3/conv2d_transpose/ReadVariableOp"
  input: "conv2d_transpose_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_transpose_3/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_transpose_3/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_transpose_3/conv2d_transpose"
  input: "conv2d_transpose_3/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_transpose_3/Relu"
  op: "Relu"
  input: "conv2d_transpose_3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000\007\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.17025130987167358
      }
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.17025130987167358
      }
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv2d_3/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv2d_3/kernel/Initializer/random_uniform/max"
  input: "conv2d_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv2d_3/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv2d_3/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_3/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv2d_3/kernel/Initializer/random_uniform/mul"
  input: "conv2d_3/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
}
node {
  name: "conv2d_3/kernel"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_3/kernel"
    }
  }
}
node {
  name: "conv2d_3/kernel/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_3/kernel"
}
node {
  name: "conv2d_3/kernel/Assign"
  op: "AssignVariableOp"
  input: "conv2d_3/kernel"
  input: "conv2d_3/kernel/Initializer/random_uniform"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/kernel/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_3/kernel"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 7
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv2d_3/bias"
  op: "VarHandleOp"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "conv2d_3/bias"
    }
  }
}
node {
  name: "conv2d_3/bias/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d_3/bias"
}
node {
  name: "conv2d_3/bias/Assign"
  op: "AssignVariableOp"
  input: "conv2d_3/bias"
  input: "conv2d_3/bias/Initializer/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/bias/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_3/bias"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "conv2d_3/Conv2D/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_3/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/Conv2D"
  op: "Conv2D"
  input: "conv2d_transpose_3/Relu"
  input: "conv2d_3/Conv2D/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv2d_3/BiasAdd/ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d_3/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3/BiasAdd"
  op: "BiasAdd"
  input: "conv2d_3/Conv2D"
  input: "conv2d_3/BiasAdd/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv2d_3/Relu"
  op: "Relu"
  input: "conv2d_3/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp"
  op: "AssignVariableOp"
  input: "conv2d/kernel"
  input: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp"
  op: "ReadVariableOp"
  input: "conv2d/kernel"
  input: "^AssignVariableOp"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_1"
  op: "AssignVariableOp"
  input: "conv2d/bias"
  input: "Placeholder_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "conv2d/bias"
  input: "^AssignVariableOp_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_2"
  op: "AssignVariableOp"
  input: "conv2d_1/kernel"
  input: "Placeholder_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_2"
  op: "ReadVariableOp"
  input: "conv2d_1/kernel"
  input: "^AssignVariableOp_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_3"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_3"
  op: "AssignVariableOp"
  input: "conv2d_1/bias"
  input: "Placeholder_3"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_3"
  op: "ReadVariableOp"
  input: "conv2d_1/bias"
  input: "^AssignVariableOp_3"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_4"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_4"
  op: "AssignVariableOp"
  input: "conv2d_2/kernel"
  input: "Placeholder_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_4"
  op: "ReadVariableOp"
  input: "conv2d_2/kernel"
  input: "^AssignVariableOp_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_5"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_5"
  op: "AssignVariableOp"
  input: "conv2d_2/bias"
  input: "Placeholder_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_5"
  op: "ReadVariableOp"
  input: "conv2d_2/bias"
  input: "^AssignVariableOp_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_6"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_6"
  op: "AssignVariableOp"
  input: "conv2d_transpose/kernel"
  input: "Placeholder_6"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_6"
  op: "ReadVariableOp"
  input: "conv2d_transpose/kernel"
  input: "^AssignVariableOp_6"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_7"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_7"
  op: "AssignVariableOp"
  input: "conv2d_transpose/bias"
  input: "Placeholder_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_7"
  op: "ReadVariableOp"
  input: "conv2d_transpose/bias"
  input: "^AssignVariableOp_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_8"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_8"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "Placeholder_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_8"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "^AssignVariableOp_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_9"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_9"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "Placeholder_9"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_9"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "^AssignVariableOp_9"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_10"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_10"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "Placeholder_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_10"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "^AssignVariableOp_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_11"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_11"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "Placeholder_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_11"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "^AssignVariableOp_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_12"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_12"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "Placeholder_12"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_12"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "^AssignVariableOp_12"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_13"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_13"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "Placeholder_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_13"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "^AssignVariableOp_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_14"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_14"
  op: "AssignVariableOp"
  input: "conv2d_3/kernel"
  input: "Placeholder_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_14"
  op: "ReadVariableOp"
  input: "conv2d_3/kernel"
  input: "^AssignVariableOp_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_15"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_15"
  op: "AssignVariableOp"
  input: "conv2d_3/bias"
  input: "Placeholder_15"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_15"
  op: "ReadVariableOp"
  input: "conv2d_3/bias"
  input: "^AssignVariableOp_15"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "conv2d/kernel"
}
node {
  name: "VarIsInitializedOp_1"
  op: "VarIsInitializedOp"
  input: "conv2d/bias"
}
node {
  name: "VarIsInitializedOp_2"
  op: "VarIsInitializedOp"
  input: "conv2d_1/kernel"
}
node {
  name: "VarIsInitializedOp_3"
  op: "VarIsInitializedOp"
  input: "conv2d_1/bias"
}
node {
  name: "VarIsInitializedOp_4"
  op: "VarIsInitializedOp"
  input: "conv2d_2/kernel"
}
node {
  name: "VarIsInitializedOp_5"
  op: "VarIsInitializedOp"
  input: "conv2d_2/bias"
}
node {
  name: "VarIsInitializedOp_6"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose/kernel"
}
node {
  name: "VarIsInitializedOp_7"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose/bias"
}
node {
  name: "VarIsInitializedOp_8"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_1/kernel"
}
node {
  name: "VarIsInitializedOp_9"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_1/bias"
}
node {
  name: "VarIsInitializedOp_10"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_2/kernel"
}
node {
  name: "VarIsInitializedOp_11"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_2/bias"
}
node {
  name: "VarIsInitializedOp_12"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_3/kernel"
}
node {
  name: "VarIsInitializedOp_13"
  op: "VarIsInitializedOp"
  input: "conv2d_transpose_3/bias"
}
node {
  name: "VarIsInitializedOp_14"
  op: "VarIsInitializedOp"
  input: "conv2d_3/kernel"
}
node {
  name: "VarIsInitializedOp_15"
  op: "VarIsInitializedOp"
  input: "conv2d_3/bias"
}
node {
  name: "init"
  op: "NoOp"
  input: "^conv2d/bias/Assign"
  input: "^conv2d/kernel/Assign"
  input: "^conv2d_1/bias/Assign"
  input: "^conv2d_1/kernel/Assign"
  input: "^conv2d_2/bias/Assign"
  input: "^conv2d_2/kernel/Assign"
  input: "^conv2d_3/bias/Assign"
  input: "^conv2d_3/kernel/Assign"
  input: "^conv2d_transpose/bias/Assign"
  input: "^conv2d_transpose/kernel/Assign"
  input: "^conv2d_transpose_1/bias/Assign"
  input: "^conv2d_transpose_1/kernel/Assign"
  input: "^conv2d_transpose_2/bias/Assign"
  input: "^conv2d_transpose_2/kernel/Assign"
  input: "^conv2d_transpose_3/bias/Assign"
  input: "^conv2d_transpose_3/kernel/Assign"
}
node {
  name: "Adam/iterations/Initializer/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT64
        tensor_shape {
        }
        int64_val: 0
      }
    }
  }
}
node {
  name: "Adam/iterations"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "Adam/iterations"
    }
  }
}
node {
  name: "Adam/iterations/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "Adam/iterations"
}
node {
  name: "Adam/iterations/Assign"
  op: "AssignVariableOp"
  input: "Adam/iterations"
  input: "Adam/iterations/Initializer/initial_value"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/iterations"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Adam/iterations/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/iterations"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/iterations"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Adam/lr/Initializer/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "Adam/lr"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "Adam/lr"
    }
  }
}
node {
  name: "Adam/lr/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "Adam/lr"
}
node {
  name: "Adam/lr/Assign"
  op: "AssignVariableOp"
  input: "Adam/lr"
  input: "Adam/lr/Initializer/initial_value"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/lr"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/lr/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/lr"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/lr"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/beta_1/Initializer/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.8999999761581421
      }
    }
  }
}
node {
  name: "Adam/beta_1"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "Adam/beta_1"
    }
  }
}
node {
  name: "Adam/beta_1/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "Adam/beta_1"
}
node {
  name: "Adam/beta_1/Assign"
  op: "AssignVariableOp"
  input: "Adam/beta_1"
  input: "Adam/beta_1/Initializer/initial_value"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/beta_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/beta_1/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/beta_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/beta_2/Initializer/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.9990000128746033
      }
    }
  }
}
node {
  name: "Adam/beta_2"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "Adam/beta_2"
    }
  }
}
node {
  name: "Adam/beta_2/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "Adam/beta_2"
}
node {
  name: "Adam/beta_2/Assign"
  op: "AssignVariableOp"
  input: "Adam/beta_2"
  input: "Adam/beta_2/Initializer/initial_value"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/beta_2"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/beta_2/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/beta_2"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/decay/Initializer/initial_value"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "Adam/decay"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "Adam/decay"
    }
  }
}
node {
  name: "Adam/decay/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "Adam/decay"
}
node {
  name: "Adam/decay/Assign"
  op: "AssignVariableOp"
  input: "Adam/decay"
  input: "Adam/decay/Initializer/initial_value"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/decay"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Adam/decay/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/decay"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@Adam/decay"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "conv2d_3_target"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: -1
        }
        dim {
          size: -1
        }
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "conv2d_3_sample_weights/input"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "conv2d_3_sample_weights"
  op: "PlaceholderWithDefault"
  input: "conv2d_3_sample_weights/input"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/sub"
  op: "Sub"
  input: "conv2d_3/Relu"
  input: "conv2d_3_target"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Square"
  op: "Square"
  input: "loss/conv2d_3_loss/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean"
  op: "Mean"
  input: "loss/conv2d_3_loss/Square"
  input: "loss/conv2d_3_loss/Mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean_1/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean_1"
  op: "Mean"
  input: "loss/conv2d_3_loss/Mean"
  input: "loss/conv2d_3_loss/Mean_1/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "loss/conv2d_3_loss/mul"
  op: "Mul"
  input: "loss/conv2d_3_loss/Mean_1"
  input: "conv2d_3_sample_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/conv2d_3_loss/NotEqual/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/NotEqual"
  op: "NotEqual"
  input: "conv2d_3_sample_weights"
  input: "loss/conv2d_3_loss/NotEqual/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Cast"
  op: "Cast"
  input: "loss/conv2d_3_loss/NotEqual"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean_2"
  op: "Mean"
  input: "loss/conv2d_3_loss/Cast"
  input: "loss/conv2d_3_loss/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "loss/conv2d_3_loss/truediv"
  op: "RealDiv"
  input: "loss/conv2d_3_loss/mul"
  input: "loss/conv2d_3_loss/Mean_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "loss/conv2d_3_loss/Mean_3"
  op: "Mean"
  input: "loss/conv2d_3_loss/truediv"
  input: "loss/conv2d_3_loss/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "loss/mul/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "loss/mul"
  op: "Mul"
  input: "loss/mul/x"
  input: "loss/conv2d_3_loss/Mean_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "metrics/mean_absolute_error/sub"
  op: "Sub"
  input: "conv2d_3/Relu"
  input: "conv2d_3_target"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "metrics/mean_absolute_error/Abs"
  op: "Abs"
  input: "metrics/mean_absolute_error/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "metrics/mean_absolute_error/Mean/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "metrics/mean_absolute_error/Mean"
  op: "Mean"
  input: "metrics/mean_absolute_error/Abs"
  input: "metrics/mean_absolute_error/Mean/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "metrics/mean_absolute_error/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "metrics/mean_absolute_error/Mean_1"
  op: "Mean"
  input: "metrics/mean_absolute_error/Mean"
  input: "metrics/mean_absolute_error/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/Shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/mul"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "training/Adam/gradients/grad_ys_0"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/mul"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/Fill"
  op: "Fill"
  input: "training/Adam/gradients/Shape"
  input: "training/Adam/gradients/grad_ys_0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/mul"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/mul_grad/Mul"
  op: "Mul"
  input: "training/Adam/gradients/Fill"
  input: "loss/conv2d_3_loss/Mean_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/mul_grad/Mul_1"
  op: "Mul"
  input: "training/Adam/gradients/Fill"
  input: "loss/mul/x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/mul_grad/Mul_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape"
  op: "Shape"
  input: "loss/conv2d_3_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Tile"
  op: "Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1"
  op: "Shape"
  input: "loss/conv2d_3_loss/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Shape_2"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/y"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum"
  op: "Maximum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordiv"
  op: "FloorDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Cast"
  op: "Cast"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truediv"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_3"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape"
  op: "Shape"
  input: "loss/conv2d_3_loss/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truediv"
  input: "loss/conv2d_3_loss/Mean_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Neg"
  op: "Neg"
  input: "loss/conv2d_3_loss/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Neg"
  input: "loss/conv2d_3_loss/Mean_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_1"
  input: "loss/conv2d_3_loss/Mean_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mul"
  op: "Mul"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_3_grad/truediv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/RealDiv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/mul"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape_1"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Sum_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/truediv"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape"
  op: "Shape"
  input: "loss/conv2d_3_loss/Mean_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1"
  op: "Shape"
  input: "conv2d_3_sample_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul"
  op: "Mul"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape"
  input: "conv2d_3_sample_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1"
  op: "Mul"
  input: "loss/conv2d_3_loss/Mean_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/truediv_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Mul_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Reshape_1"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Sum_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/mul"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape"
  op: "Shape"
  input: "loss/conv2d_3_loss/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/add"
  op: "Add"
  input: "loss/conv2d_3_loss/Mean_1/reduction_indices"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/mod"
  op: "FloorMod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/add"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/start"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/delta"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range"
  op: "Range"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/start"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Size"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill"
  op: "Fill"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill/value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch"
  op: "DynamicStitch"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/range"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/mod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Fill"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/y"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum"
  op: "Maximum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv"
  op: "FloorDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/mul_grad/Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/DynamicStitch"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Tile"
  op: "Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2"
  op: "Shape"
  input: "loss/conv2d_3_loss/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3"
  op: "Shape"
  input: "loss/conv2d_3_loss/Mean_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_2"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Shape_3"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/y"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1"
  op: "Maximum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1"
  op: "FloorDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Maximum_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Cast"
  op: "Cast"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/floordiv_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truediv"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean_1"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape"
  op: "Shape"
  input: "loss/conv2d_3_loss/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/add"
  op: "Add"
  input: "loss/conv2d_3_loss/Mean/reduction_indices"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/mod"
  op: "FloorMod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/add"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/start"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/delta"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range"
  op: "Range"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/start"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Size"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill"
  op: "Fill"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill/value"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch"
  op: "DynamicStitch"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/range"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/mod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Fill"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum"
  op: "Maximum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv"
  op: "FloorDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_1_grad/truediv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/DynamicStitch"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Tile"
  op: "Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2"
  op: "Shape"
  input: "loss/conv2d_3_loss/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3"
  op: "Shape"
  input: "loss/conv2d_3_loss/Mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_2"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1"
  op: "Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Shape_3"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/y"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1"
  op: "Maximum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1"
  op: "FloorDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Prod"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Maximum_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Cast"
  op: "Cast"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/floordiv_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truediv"
  op: "RealDiv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Tile"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Mean"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Const"
  op: "Const"
  input: "^training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truediv"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Square"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Mul"
  op: "Mul"
  input: "loss/conv2d_3_loss/sub"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Square"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Mul_1"
  op: "Mul"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Mean_grad/truediv"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/Square"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape"
  op: "Shape"
  input: "conv2d_3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1"
  op: "Shape"
  input: "conv2d_3_target"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Mul_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1"
  op: "Sum"
  input: "training/Adam/gradients/loss/conv2d_3_loss/Square_grad/Mul_1"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Neg"
  op: "Neg"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Sum_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape_1"
  op: "Reshape"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Neg"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@loss/conv2d_3_loss/sub"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/loss/conv2d_3_loss/sub_grad/Reshape"
  input: "conv2d_3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_3/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "conv2d_transpose_3/Relu"
  input: "conv2d_3/Conv2D/ReadVariableOp"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/Conv2D"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "training/Adam/gradients/conv2d_3/Conv2D_grad/ShapeN"
  input: "conv2d_3/Conv2D/ReadVariableOp"
  input: "training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "conv2d_transpose_3/Relu"
  input: "training/Adam/gradients/conv2d_3/Conv2D_grad/ShapeN:1"
  input: "training/Adam/gradients/conv2d_3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_3/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_3/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput"
  input: "conv2d_transpose_3/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_3/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_transpose_3/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/conv2d_transpose"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "training/Adam/gradients/conv2d_transpose_3/Relu_grad/ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Shape"
  input: "conv2d_transpose_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Conv2D"
  op: "Conv2D"
  input: "training/Adam/gradients/conv2d_transpose_3/Relu_grad/ReluGrad"
  input: "conv2d_transpose_3/conv2d_transpose/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_3/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Conv2D"
  input: "conv2d_transpose_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_transpose_2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/conv2d_transpose"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "training/Adam/gradients/conv2d_transpose_2/Relu_grad/ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Shape"
  input: "conv2d_transpose_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Conv2D"
  op: "Conv2D"
  input: "training/Adam/gradients/conv2d_transpose_2/Relu_grad/ReluGrad"
  input: "conv2d_transpose_2/conv2d_transpose/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_2/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Conv2D"
  input: "conv2d_transpose_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_transpose_1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/conv2d_transpose"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "training/Adam/gradients/conv2d_transpose_1/Relu_grad/ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Shape"
  input: "conv2d_transpose/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2D"
  op: "Conv2D"
  input: "training/Adam/gradients/conv2d_transpose_1/Relu_grad/ReluGrad"
  input: "conv2d_transpose_1/conv2d_transpose/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose_1/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2D"
  input: "conv2d_transpose/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_transpose/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/conv2d_transpose"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "training/Adam/gradients/conv2d_transpose/Relu_grad/ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Shape"
  input: "conv2d_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Conv2D"
  op: "Conv2D"
  input: "training/Adam/gradients/conv2d_transpose/Relu_grad/ReluGrad"
  input: "conv2d_transpose/conv2d_transpose/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_transpose/conv2d_transpose"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Conv2D"
  input: "conv2d_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/BatchToSpaceND_grad/SpaceToBatchND"
  op: "SpaceToBatchND"
  input: "training/Adam/gradients/conv2d_2/Relu_grad/ReluGrad"
  input: "conv2d_2/BatchToSpaceND/block_shape"
  input: "conv2d_2/BatchToSpaceND/crops"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/BatchToSpaceND"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "conv2d_2/SpaceToBatchND"
  input: "conv2d_2/Conv2D/ReadVariableOp"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/Conv2D"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "training/Adam/gradients/conv2d_2/Conv2D_grad/ShapeN"
  input: "conv2d_2/Conv2D/ReadVariableOp"
  input: "training/Adam/gradients/conv2d_2/BatchToSpaceND_grad/SpaceToBatchND"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "conv2d_2/SpaceToBatchND"
  input: "training/Adam/gradients/conv2d_2/Conv2D_grad/ShapeN:1"
  input: "training/Adam/gradients/conv2d_2/BatchToSpaceND_grad/SpaceToBatchND"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_2/SpaceToBatchND_grad/BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "training/Adam/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput"
  input: "conv2d_2/SpaceToBatchND/block_shape"
  input: "conv2d_2/SpaceToBatchND/paddings"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tcrops"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_2/SpaceToBatchND"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_2/SpaceToBatchND_grad/BatchToSpaceND"
  input: "conv2d_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/BatchToSpaceND_grad/SpaceToBatchND"
  op: "SpaceToBatchND"
  input: "training/Adam/gradients/conv2d_1/Relu_grad/ReluGrad"
  input: "conv2d_1/BatchToSpaceND/block_shape"
  input: "conv2d_1/BatchToSpaceND/crops"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/BatchToSpaceND"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "conv2d_1/SpaceToBatchND"
  input: "conv2d_1/Conv2D/ReadVariableOp"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/Conv2D"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeN"
  input: "conv2d_1/Conv2D/ReadVariableOp"
  input: "training/Adam/gradients/conv2d_1/BatchToSpaceND_grad/SpaceToBatchND"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "conv2d_1/SpaceToBatchND"
  input: "training/Adam/gradients/conv2d_1/Conv2D_grad/ShapeN:1"
  input: "training/Adam/gradients/conv2d_1/BatchToSpaceND_grad/SpaceToBatchND"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d_1/SpaceToBatchND_grad/BatchToSpaceND"
  op: "BatchToSpaceND"
  input: "training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput"
  input: "conv2d_1/SpaceToBatchND/block_shape"
  input: "conv2d_1/SpaceToBatchND/paddings"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tblock_shape"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tcrops"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d_1/SpaceToBatchND"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "training/Adam/gradients/conv2d_1/SpaceToBatchND_grad/BatchToSpaceND"
  input: "conv2d/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/Relu"
      }
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "training/Adam/gradients/conv2d/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/BiasAdd"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d/Conv2D_grad/ShapeN"
  op: "ShapeN"
  input: "ae_input"
  input: "conv2d/Conv2D/ReadVariableOp"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/Conv2D"
      }
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "training/Adam/gradients/conv2d/Conv2D_grad/ShapeN"
  input: "conv2d/Conv2D/ReadVariableOp"
  input: "training/Adam/gradients/conv2d/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "ae_input"
  input: "training/Adam/gradients/conv2d/Conv2D_grad/ShapeN:1"
  input: "training/Adam/gradients/conv2d/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv2d/Conv2D"
      }
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "training/Adam/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT64
        tensor_shape {
        }
        int64_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/AssignAddVariableOp"
  op: "AssignAddVariableOp"
  input: "Adam/iterations"
  input: "training/Adam/Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/iterations"
  input: "^training/Adam/AssignAddVariableOp"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "training/Adam/Cast/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/iterations"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "training/Adam/Cast"
  op: "Cast"
  input: "training/Adam/Cast/ReadVariableOp"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "training/Adam/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/add"
  op: "Add"
  input: "training/Adam/Cast"
  input: "training/Adam/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Pow/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Pow"
  op: "Pow"
  input: "training/Adam/Pow/ReadVariableOp"
  input: "training/Adam/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub"
  op: "Sub"
  input: "training/Adam/sub/x"
  input: "training/Adam/Pow"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value/Minimum"
  op: "Minimum"
  input: "training/Adam/sub"
  input: "training/Adam/Const_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value"
  op: "Maximum"
  input: "training/Adam/clip_by_value/Minimum"
  input: "training/Adam/Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt"
  op: "Sqrt"
  input: "training/Adam/clip_by_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Pow_1/ReadVariableOp"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Pow_1"
  op: "Pow"
  input: "training/Adam/Pow_1/ReadVariableOp"
  input: "training/Adam/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_1/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_1"
  op: "Sub"
  input: "training/Adam/sub_1/x"
  input: "training/Adam/Pow_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv"
  op: "RealDiv"
  input: "training/Adam/Sqrt"
  input: "training/Adam/sub_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_1"
  op: "ReadVariableOp"
  input: "Adam/lr"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_1"
  input: "training/Adam/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\007\000\000\000\020\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros"
  op: "Fill"
  input: "training/Adam/zeros/shape_as_tensor"
  input: "training/Adam/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable"
    }
  }
}
node {
  name: "training/Adam/Variable/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable"
}
node {
  name: "training/Adam/Variable/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable"
  input: "training/Adam/zeros"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_1"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_1"
    }
  }
}
node {
  name: "training/Adam/Variable_1/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_1"
}
node {
  name: "training/Adam/Variable_1/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_1"
  input: "training/Adam/zeros_1"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_1/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_1"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_2/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_2/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_2"
  op: "Fill"
  input: "training/Adam/zeros_2/shape_as_tensor"
  input: "training/Adam/zeros_2/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_2"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_2"
    }
  }
}
node {
  name: "training/Adam/Variable_2/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_2"
}
node {
  name: "training/Adam/Variable_2/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_2"
  input: "training/Adam/zeros_2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_2"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_2/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_2"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_3"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_3"
    }
  }
}
node {
  name: "training/Adam/Variable_3/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_3"
}
node {
  name: "training/Adam/Variable_3/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_3"
  input: "training/Adam/zeros_3"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_3/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_3"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_3"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_4/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_4/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_4"
  op: "Fill"
  input: "training/Adam/zeros_4/shape_as_tensor"
  input: "training/Adam/zeros_4/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_4"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_4"
    }
  }
}
node {
  name: "training/Adam/Variable_4/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_4"
}
node {
  name: "training/Adam/Variable_4/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_4"
  input: "training/Adam/zeros_4"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_4"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_4/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_4"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_4"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_5"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_5"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_5"
    }
  }
}
node {
  name: "training/Adam/Variable_5/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_5"
}
node {
  name: "training/Adam/Variable_5/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_5"
  input: "training/Adam/zeros_5"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_5"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_5/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_5"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_5"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_6/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_6/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_6"
  op: "Fill"
  input: "training/Adam/zeros_6/shape_as_tensor"
  input: "training/Adam/zeros_6/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_6"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_6"
    }
  }
}
node {
  name: "training/Adam/Variable_6/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_6"
}
node {
  name: "training/Adam/Variable_6/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_6"
  input: "training/Adam/zeros_6"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_6"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_6/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_6"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_6"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_7"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_7"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_7"
    }
  }
}
node {
  name: "training/Adam/Variable_7/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_7"
}
node {
  name: "training/Adam/Variable_7/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_7"
  input: "training/Adam/zeros_7"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_7"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_7/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_7"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_7"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_8/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_8/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_8"
  op: "Fill"
  input: "training/Adam/zeros_8/shape_as_tensor"
  input: "training/Adam/zeros_8/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_8"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_8"
    }
  }
}
node {
  name: "training/Adam/Variable_8/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_8"
}
node {
  name: "training/Adam/Variable_8/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_8"
  input: "training/Adam/zeros_8"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_8"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_8/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_8"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_8"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_9"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_9"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_9"
    }
  }
}
node {
  name: "training/Adam/Variable_9/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_9"
}
node {
  name: "training/Adam/Variable_9/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_9"
  input: "training/Adam/zeros_9"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_9"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_9/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_9"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_9"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_10/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_10/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_10"
  op: "Fill"
  input: "training/Adam/zeros_10/shape_as_tensor"
  input: "training/Adam/zeros_10/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_10"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_10"
    }
  }
}
node {
  name: "training/Adam/Variable_10/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_10"
}
node {
  name: "training/Adam/Variable_10/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_10"
  input: "training/Adam/zeros_10"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_10"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_10/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_10"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_10"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_11"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_11"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_11"
    }
  }
}
node {
  name: "training/Adam/Variable_11/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_11"
}
node {
  name: "training/Adam/Variable_11/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_11"
  input: "training/Adam/zeros_11"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_11"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_11/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_11"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_11"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_12/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_12/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_12"
  op: "Fill"
  input: "training/Adam/zeros_12/shape_as_tensor"
  input: "training/Adam/zeros_12/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_12"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_12"
    }
  }
}
node {
  name: "training/Adam/Variable_12/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_12"
}
node {
  name: "training/Adam/Variable_12/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_12"
  input: "training/Adam/zeros_12"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_12"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_12/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_12"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_12"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_13"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_13"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_13"
    }
  }
}
node {
  name: "training/Adam/Variable_13/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_13"
}
node {
  name: "training/Adam/Variable_13/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_13"
  input: "training/Adam/zeros_13"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_13"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_13/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_13"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_13"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_14/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000\007\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_14/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_14"
  op: "Fill"
  input: "training/Adam/zeros_14/shape_as_tensor"
  input: "training/Adam/zeros_14/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_14"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_14"
    }
  }
}
node {
  name: "training/Adam/Variable_14/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_14"
}
node {
  name: "training/Adam/Variable_14/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_14"
  input: "training/Adam/zeros_14"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_14"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_14/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_14"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_14"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_15"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 7
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_15"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_15"
    }
  }
}
node {
  name: "training/Adam/Variable_15/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_15"
}
node {
  name: "training/Adam/Variable_15/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_15"
  input: "training/Adam/zeros_15"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_15"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_15/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_15"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_15"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_16/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\007\000\000\000\020\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_16/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_16"
  op: "Fill"
  input: "training/Adam/zeros_16/shape_as_tensor"
  input: "training/Adam/zeros_16/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_16"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_16"
    }
  }
}
node {
  name: "training/Adam/Variable_16/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_16"
}
node {
  name: "training/Adam/Variable_16/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_16"
  input: "training/Adam/zeros_16"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_16"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_16/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_16"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_16"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_17"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_17"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_17"
    }
  }
}
node {
  name: "training/Adam/Variable_17/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_17"
}
node {
  name: "training/Adam/Variable_17/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_17"
  input: "training/Adam/zeros_17"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_17"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_17/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_17"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_17"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_18/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_18/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_18"
  op: "Fill"
  input: "training/Adam/zeros_18/shape_as_tensor"
  input: "training/Adam/zeros_18/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_18"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_18"
    }
  }
}
node {
  name: "training/Adam/Variable_18/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_18"
}
node {
  name: "training/Adam/Variable_18/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_18"
  input: "training/Adam/zeros_18"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_18"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_18/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_18"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_18"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_19"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_19"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_19"
    }
  }
}
node {
  name: "training/Adam/Variable_19/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_19"
}
node {
  name: "training/Adam/Variable_19/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_19"
  input: "training/Adam/zeros_19"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_19"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_19/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_19"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_19"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_20/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_20/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_20"
  op: "Fill"
  input: "training/Adam/zeros_20/shape_as_tensor"
  input: "training/Adam/zeros_20/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_20"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_20"
    }
  }
}
node {
  name: "training/Adam/Variable_20/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_20"
}
node {
  name: "training/Adam/Variable_20/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_20"
  input: "training/Adam/zeros_20"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_20"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_20/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_20"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_20"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_21"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_21"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_21"
    }
  }
}
node {
  name: "training/Adam/Variable_21/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_21"
}
node {
  name: "training/Adam/Variable_21/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_21"
  input: "training/Adam/zeros_21"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_21"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_21/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_21"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_21"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_22/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_22/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_22"
  op: "Fill"
  input: "training/Adam/zeros_22/shape_as_tensor"
  input: "training/Adam/zeros_22/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_22"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_22"
    }
  }
}
node {
  name: "training/Adam/Variable_22/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_22"
}
node {
  name: "training/Adam/Variable_22/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_22"
  input: "training/Adam/zeros_22"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_22"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_22/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_22"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_22"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_23"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 128
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_23"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_23"
    }
  }
}
node {
  name: "training/Adam/Variable_23/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_23"
}
node {
  name: "training/Adam/Variable_23/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_23"
  input: "training/Adam/zeros_23"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_23"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_23/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_23"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_23"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_24/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000\200\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_24/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_24"
  op: "Fill"
  input: "training/Adam/zeros_24/shape_as_tensor"
  input: "training/Adam/zeros_24/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_24"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_24"
    }
  }
}
node {
  name: "training/Adam/Variable_24/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_24"
}
node {
  name: "training/Adam/Variable_24/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_24"
  input: "training/Adam/zeros_24"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_24"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_24/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_24"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_24"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_25"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_25"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_25"
    }
  }
}
node {
  name: "training/Adam/Variable_25/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_25"
}
node {
  name: "training/Adam/Variable_25/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_25"
  input: "training/Adam/zeros_25"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_25"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_25/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_25"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_25"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_26/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_26/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_26"
  op: "Fill"
  input: "training/Adam/zeros_26/shape_as_tensor"
  input: "training/Adam/zeros_26/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_26"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_26"
    }
  }
}
node {
  name: "training/Adam/Variable_26/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_26"
}
node {
  name: "training/Adam/Variable_26/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_26"
  input: "training/Adam/zeros_26"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_26"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_26/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_26"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_26"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_27"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_27"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_27"
    }
  }
}
node {
  name: "training/Adam/Variable_27/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_27"
}
node {
  name: "training/Adam/Variable_27/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_27"
  input: "training/Adam/zeros_27"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_27"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_27/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_27"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_27"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_28/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_28/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_28"
  op: "Fill"
  input: "training/Adam/zeros_28/shape_as_tensor"
  input: "training/Adam/zeros_28/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_28"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_28"
    }
  }
}
node {
  name: "training/Adam/Variable_28/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_28"
}
node {
  name: "training/Adam/Variable_28/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_28"
  input: "training/Adam/zeros_28"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_28"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_28/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_28"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_28"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_29"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_29"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_29"
    }
  }
}
node {
  name: "training/Adam/Variable_29/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_29"
}
node {
  name: "training/Adam/Variable_29/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_29"
  input: "training/Adam/zeros_29"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_29"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_29/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_29"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_29"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_30/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000\007\000\000\000"
      }
    }
  }
}
node {
  name: "training/Adam/zeros_30/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_30"
  op: "Fill"
  input: "training/Adam/zeros_30/shape_as_tensor"
  input: "training/Adam/zeros_30/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_30"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_30"
    }
  }
}
node {
  name: "training/Adam/Variable_30/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_30"
}
node {
  name: "training/Adam/Variable_30/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_30"
  input: "training/Adam/zeros_30"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_30"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_30/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_30"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_30"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_31"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 7
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Variable_31"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_31"
    }
  }
}
node {
  name: "training/Adam/Variable_31/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_31"
}
node {
  name: "training/Adam/Variable_31/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_31"
  input: "training/Adam/zeros_31"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_31"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_31/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_31"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_31"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_32/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_32/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_32"
  op: "Fill"
  input: "training/Adam/zeros_32/shape_as_tensor"
  input: "training/Adam/zeros_32/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_32"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_32"
    }
  }
}
node {
  name: "training/Adam/Variable_32/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_32"
}
node {
  name: "training/Adam/Variable_32/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_32"
  input: "training/Adam/zeros_32"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_32"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_32/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_32"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_32"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_33/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_33/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_33"
  op: "Fill"
  input: "training/Adam/zeros_33/shape_as_tensor"
  input: "training/Adam/zeros_33/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_33"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_33"
    }
  }
}
node {
  name: "training/Adam/Variable_33/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_33"
}
node {
  name: "training/Adam/Variable_33/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_33"
  input: "training/Adam/zeros_33"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_33"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_33/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_33"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_33"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_34/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_34/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_34"
  op: "Fill"
  input: "training/Adam/zeros_34/shape_as_tensor"
  input: "training/Adam/zeros_34/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_34"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_34"
    }
  }
}
node {
  name: "training/Adam/Variable_34/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_34"
}
node {
  name: "training/Adam/Variable_34/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_34"
  input: "training/Adam/zeros_34"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_34"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_34/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_34"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_34"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_35/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_35/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_35"
  op: "Fill"
  input: "training/Adam/zeros_35/shape_as_tensor"
  input: "training/Adam/zeros_35/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_35"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_35"
    }
  }
}
node {
  name: "training/Adam/Variable_35/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_35"
}
node {
  name: "training/Adam/Variable_35/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_35"
  input: "training/Adam/zeros_35"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_35"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_35/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_35"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_35"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_36/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_36/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_36"
  op: "Fill"
  input: "training/Adam/zeros_36/shape_as_tensor"
  input: "training/Adam/zeros_36/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_36"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_36"
    }
  }
}
node {
  name: "training/Adam/Variable_36/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_36"
}
node {
  name: "training/Adam/Variable_36/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_36"
  input: "training/Adam/zeros_36"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_36"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_36/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_36"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_36"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_37/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_37/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_37"
  op: "Fill"
  input: "training/Adam/zeros_37/shape_as_tensor"
  input: "training/Adam/zeros_37/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_37"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_37"
    }
  }
}
node {
  name: "training/Adam/Variable_37/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_37"
}
node {
  name: "training/Adam/Variable_37/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_37"
  input: "training/Adam/zeros_37"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_37"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_37/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_37"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_37"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_38/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_38/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_38"
  op: "Fill"
  input: "training/Adam/zeros_38/shape_as_tensor"
  input: "training/Adam/zeros_38/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_38"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_38"
    }
  }
}
node {
  name: "training/Adam/Variable_38/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_38"
}
node {
  name: "training/Adam/Variable_38/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_38"
  input: "training/Adam/zeros_38"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_38"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_38/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_38"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_38"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_39/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_39/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_39"
  op: "Fill"
  input: "training/Adam/zeros_39/shape_as_tensor"
  input: "training/Adam/zeros_39/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_39"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_39"
    }
  }
}
node {
  name: "training/Adam/Variable_39/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_39"
}
node {
  name: "training/Adam/Variable_39/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_39"
  input: "training/Adam/zeros_39"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_39"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_39/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_39"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_39"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_40/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_40/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_40"
  op: "Fill"
  input: "training/Adam/zeros_40/shape_as_tensor"
  input: "training/Adam/zeros_40/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_40"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_40"
    }
  }
}
node {
  name: "training/Adam/Variable_40/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_40"
}
node {
  name: "training/Adam/Variable_40/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_40"
  input: "training/Adam/zeros_40"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_40"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_40/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_40"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_40"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_41/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_41/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_41"
  op: "Fill"
  input: "training/Adam/zeros_41/shape_as_tensor"
  input: "training/Adam/zeros_41/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_41"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_41"
    }
  }
}
node {
  name: "training/Adam/Variable_41/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_41"
}
node {
  name: "training/Adam/Variable_41/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_41"
  input: "training/Adam/zeros_41"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_41"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_41/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_41"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_41"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_42/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_42/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_42"
  op: "Fill"
  input: "training/Adam/zeros_42/shape_as_tensor"
  input: "training/Adam/zeros_42/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_42"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_42"
    }
  }
}
node {
  name: "training/Adam/Variable_42/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_42"
}
node {
  name: "training/Adam/Variable_42/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_42"
  input: "training/Adam/zeros_42"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_42"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_42/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_42"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_42"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_43/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_43/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_43"
  op: "Fill"
  input: "training/Adam/zeros_43/shape_as_tensor"
  input: "training/Adam/zeros_43/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_43"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_43"
    }
  }
}
node {
  name: "training/Adam/Variable_43/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_43"
}
node {
  name: "training/Adam/Variable_43/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_43"
  input: "training/Adam/zeros_43"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_43"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_43/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_43"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_43"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_44/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_44/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_44"
  op: "Fill"
  input: "training/Adam/zeros_44/shape_as_tensor"
  input: "training/Adam/zeros_44/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_44"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_44"
    }
  }
}
node {
  name: "training/Adam/Variable_44/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_44"
}
node {
  name: "training/Adam/Variable_44/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_44"
  input: "training/Adam/zeros_44"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_44"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_44/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_44"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_44"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_45/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_45/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_45"
  op: "Fill"
  input: "training/Adam/zeros_45/shape_as_tensor"
  input: "training/Adam/zeros_45/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_45"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_45"
    }
  }
}
node {
  name: "training/Adam/Variable_45/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_45"
}
node {
  name: "training/Adam/Variable_45/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_45"
  input: "training/Adam/zeros_45"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_45"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_45/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_45"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_45"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_46/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_46/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_46"
  op: "Fill"
  input: "training/Adam/zeros_46/shape_as_tensor"
  input: "training/Adam/zeros_46/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_46"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_46"
    }
  }
}
node {
  name: "training/Adam/Variable_46/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_46"
}
node {
  name: "training/Adam/Variable_46/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_46"
  input: "training/Adam/zeros_46"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_46"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_46/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_46"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_46"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/zeros_47/shape_as_tensor"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "training/Adam/zeros_47/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/zeros_47"
  op: "Fill"
  input: "training/Adam/zeros_47/shape_as_tensor"
  input: "training/Adam/zeros_47/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "training/Adam/Variable_47"
  op: "VarHandleOp"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: "training/Adam/Variable_47"
    }
  }
}
node {
  name: "training/Adam/Variable_47/IsInitialized/VarIsInitializedOp"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_47"
}
node {
  name: "training/Adam/Variable_47/Assign"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_47"
  input: "training/Adam/zeros_47"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_47"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Variable_47/Read/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_47"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@training/Adam/Variable_47"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_2"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_1/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_1"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_2"
  input: "training/Adam/mul_1/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_3"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_2/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_2"
  op: "Sub"
  input: "training/Adam/sub_2/x"
  input: "training/Adam/ReadVariableOp_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_2"
  op: "Mul"
  input: "training/Adam/sub_2"
  input: "training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_1"
  op: "Add"
  input: "training/Adam/mul_1"
  input: "training/Adam/mul_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_4"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_3/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_16"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_3"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_4"
  input: "training/Adam/mul_3/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_5"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_3/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_3"
  op: "Sub"
  input: "training/Adam/sub_3/x"
  input: "training/Adam/ReadVariableOp_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square"
  op: "Square"
  input: "training/Adam/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_4"
  op: "Mul"
  input: "training/Adam/sub_3"
  input: "training/Adam/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_2"
  op: "Add"
  input: "training/Adam/mul_3"
  input: "training/Adam/mul_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_5"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_4"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_1/Minimum"
  op: "Minimum"
  input: "training/Adam/add_2"
  input: "training/Adam/Const_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_1"
  op: "Maximum"
  input: "training/Adam/clip_by_value_1/Minimum"
  input: "training/Adam/Const_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_1"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_3/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_3"
  op: "Add"
  input: "training/Adam/Sqrt_1"
  input: "training/Adam/add_3/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_1"
  op: "RealDiv"
  input: "training/Adam/mul_5"
  input: "training/Adam/add_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_6"
  op: "ReadVariableOp"
  input: "conv2d/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_4"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_6"
  input: "training/Adam/truediv_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp"
  op: "AssignVariableOp"
  input: "training/Adam/Variable"
  input: "training/Adam/add_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_7"
  op: "ReadVariableOp"
  input: "training/Adam/Variable"
  input: "^training/Adam/AssignVariableOp"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_1"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_16"
  input: "training/Adam/add_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_8"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_16"
  input: "^training/Adam/AssignVariableOp_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_2"
  op: "AssignVariableOp"
  input: "conv2d/kernel"
  input: "training/Adam/sub_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_9"
  op: "ReadVariableOp"
  input: "conv2d/kernel"
  input: "^training/Adam/AssignVariableOp_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_10"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_6/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_6"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_10"
  input: "training/Adam/mul_6/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_11"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_5/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_5"
  op: "Sub"
  input: "training/Adam/sub_5/x"
  input: "training/Adam/ReadVariableOp_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_7"
  op: "Mul"
  input: "training/Adam/sub_5"
  input: "training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_4"
  op: "Add"
  input: "training/Adam/mul_6"
  input: "training/Adam/mul_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_12"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_8/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_8"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_12"
  input: "training/Adam/mul_8/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_13"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_6/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_6"
  op: "Sub"
  input: "training/Adam/sub_6/x"
  input: "training/Adam/ReadVariableOp_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_1"
  op: "Square"
  input: "training/Adam/gradients/conv2d/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_9"
  op: "Mul"
  input: "training/Adam/sub_6"
  input: "training/Adam/Square_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_5"
  op: "Add"
  input: "training/Adam/mul_8"
  input: "training/Adam/mul_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_10"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_5"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_6"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_2/Minimum"
  op: "Minimum"
  input: "training/Adam/add_5"
  input: "training/Adam/Const_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_2"
  op: "Maximum"
  input: "training/Adam/clip_by_value_2/Minimum"
  input: "training/Adam/Const_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_2"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_6/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_6"
  op: "Add"
  input: "training/Adam/Sqrt_2"
  input: "training/Adam/add_6/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_2"
  op: "RealDiv"
  input: "training/Adam/mul_10"
  input: "training/Adam/add_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_14"
  op: "ReadVariableOp"
  input: "conv2d/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_7"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_14"
  input: "training/Adam/truediv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_3"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_1"
  input: "training/Adam/add_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_15"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_1"
  input: "^training/Adam/AssignVariableOp_3"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_4"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_17"
  input: "training/Adam/add_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_16"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_17"
  input: "^training/Adam/AssignVariableOp_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_5"
  op: "AssignVariableOp"
  input: "conv2d/bias"
  input: "training/Adam/sub_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_17"
  op: "ReadVariableOp"
  input: "conv2d/bias"
  input: "^training/Adam/AssignVariableOp_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_18"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_11/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_11"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_18"
  input: "training/Adam/mul_11/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_19"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_8/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_8"
  op: "Sub"
  input: "training/Adam/sub_8/x"
  input: "training/Adam/ReadVariableOp_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_12"
  op: "Mul"
  input: "training/Adam/sub_8"
  input: "training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_7"
  op: "Add"
  input: "training/Adam/mul_11"
  input: "training/Adam/mul_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_20"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_13/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_18"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_13"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_20"
  input: "training/Adam/mul_13/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_21"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_9/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_9"
  op: "Sub"
  input: "training/Adam/sub_9/x"
  input: "training/Adam/ReadVariableOp_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_2"
  op: "Square"
  input: "training/Adam/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_14"
  op: "Mul"
  input: "training/Adam/sub_9"
  input: "training/Adam/Square_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_8"
  op: "Add"
  input: "training/Adam/mul_13"
  input: "training/Adam/mul_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_15"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_7"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_8"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_3/Minimum"
  op: "Minimum"
  input: "training/Adam/add_8"
  input: "training/Adam/Const_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_3"
  op: "Maximum"
  input: "training/Adam/clip_by_value_3/Minimum"
  input: "training/Adam/Const_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_3"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_9/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_9"
  op: "Add"
  input: "training/Adam/Sqrt_3"
  input: "training/Adam/add_9/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_3"
  op: "RealDiv"
  input: "training/Adam/mul_15"
  input: "training/Adam/add_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_22"
  op: "ReadVariableOp"
  input: "conv2d_1/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_10"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_22"
  input: "training/Adam/truediv_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_6"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_2"
  input: "training/Adam/add_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_23"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_2"
  input: "^training/Adam/AssignVariableOp_6"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_7"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_18"
  input: "training/Adam/add_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_24"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_18"
  input: "^training/Adam/AssignVariableOp_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_8"
  op: "AssignVariableOp"
  input: "conv2d_1/kernel"
  input: "training/Adam/sub_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_25"
  op: "ReadVariableOp"
  input: "conv2d_1/kernel"
  input: "^training/Adam/AssignVariableOp_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_26"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_16/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_3"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_16"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_26"
  input: "training/Adam/mul_16/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_27"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_11/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_11"
  op: "Sub"
  input: "training/Adam/sub_11/x"
  input: "training/Adam/ReadVariableOp_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_17"
  op: "Mul"
  input: "training/Adam/sub_11"
  input: "training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_10"
  op: "Add"
  input: "training/Adam/mul_16"
  input: "training/Adam/mul_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_28"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_18/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_18"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_28"
  input: "training/Adam/mul_18/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_29"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_12/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_12"
  op: "Sub"
  input: "training/Adam/sub_12/x"
  input: "training/Adam/ReadVariableOp_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_3"
  op: "Square"
  input: "training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_19"
  op: "Mul"
  input: "training/Adam/sub_12"
  input: "training/Adam/Square_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_11"
  op: "Add"
  input: "training/Adam/mul_18"
  input: "training/Adam/mul_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_20"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_9"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_10"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_4/Minimum"
  op: "Minimum"
  input: "training/Adam/add_11"
  input: "training/Adam/Const_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_4"
  op: "Maximum"
  input: "training/Adam/clip_by_value_4/Minimum"
  input: "training/Adam/Const_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_4"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_12/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_12"
  op: "Add"
  input: "training/Adam/Sqrt_4"
  input: "training/Adam/add_12/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_4"
  op: "RealDiv"
  input: "training/Adam/mul_20"
  input: "training/Adam/add_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_30"
  op: "ReadVariableOp"
  input: "conv2d_1/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_13"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_30"
  input: "training/Adam/truediv_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_9"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_3"
  input: "training/Adam/add_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_31"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_3"
  input: "^training/Adam/AssignVariableOp_9"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_10"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_19"
  input: "training/Adam/add_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_32"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_19"
  input: "^training/Adam/AssignVariableOp_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_11"
  op: "AssignVariableOp"
  input: "conv2d_1/bias"
  input: "training/Adam/sub_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_33"
  op: "ReadVariableOp"
  input: "conv2d_1/bias"
  input: "^training/Adam/AssignVariableOp_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_34"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_21/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_21"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_34"
  input: "training/Adam/mul_21/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_35"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_14/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_14"
  op: "Sub"
  input: "training/Adam/sub_14/x"
  input: "training/Adam/ReadVariableOp_35"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_22"
  op: "Mul"
  input: "training/Adam/sub_14"
  input: "training/Adam/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_13"
  op: "Add"
  input: "training/Adam/mul_21"
  input: "training/Adam/mul_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_36"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_23/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_23"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_36"
  input: "training/Adam/mul_23/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_37"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_15/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_15"
  op: "Sub"
  input: "training/Adam/sub_15/x"
  input: "training/Adam/ReadVariableOp_37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_4"
  op: "Square"
  input: "training/Adam/gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_24"
  op: "Mul"
  input: "training/Adam/sub_15"
  input: "training/Adam/Square_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_14"
  op: "Add"
  input: "training/Adam/mul_23"
  input: "training/Adam/mul_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_25"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_11"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_12"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_5/Minimum"
  op: "Minimum"
  input: "training/Adam/add_14"
  input: "training/Adam/Const_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_5"
  op: "Maximum"
  input: "training/Adam/clip_by_value_5/Minimum"
  input: "training/Adam/Const_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_5"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_15/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_15"
  op: "Add"
  input: "training/Adam/Sqrt_5"
  input: "training/Adam/add_15/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_5"
  op: "RealDiv"
  input: "training/Adam/mul_25"
  input: "training/Adam/add_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_38"
  op: "ReadVariableOp"
  input: "conv2d_2/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_16"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_38"
  input: "training/Adam/truediv_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_12"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_4"
  input: "training/Adam/add_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_39"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_4"
  input: "^training/Adam/AssignVariableOp_12"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_13"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_20"
  input: "training/Adam/add_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_40"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_20"
  input: "^training/Adam/AssignVariableOp_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_14"
  op: "AssignVariableOp"
  input: "conv2d_2/kernel"
  input: "training/Adam/sub_16"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_41"
  op: "ReadVariableOp"
  input: "conv2d_2/kernel"
  input: "^training/Adam/AssignVariableOp_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_42"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_26/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_26"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_42"
  input: "training/Adam/mul_26/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_43"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_17/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_17"
  op: "Sub"
  input: "training/Adam/sub_17/x"
  input: "training/Adam/ReadVariableOp_43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_27"
  op: "Mul"
  input: "training/Adam/sub_17"
  input: "training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_16"
  op: "Add"
  input: "training/Adam/mul_26"
  input: "training/Adam/mul_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_44"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_28/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_21"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_28"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_44"
  input: "training/Adam/mul_28/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_45"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_18/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_18"
  op: "Sub"
  input: "training/Adam/sub_18/x"
  input: "training/Adam/ReadVariableOp_45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_5"
  op: "Square"
  input: "training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_29"
  op: "Mul"
  input: "training/Adam/sub_18"
  input: "training/Adam/Square_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_17"
  op: "Add"
  input: "training/Adam/mul_28"
  input: "training/Adam/mul_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_30"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_13"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_14"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_6/Minimum"
  op: "Minimum"
  input: "training/Adam/add_17"
  input: "training/Adam/Const_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_6"
  op: "Maximum"
  input: "training/Adam/clip_by_value_6/Minimum"
  input: "training/Adam/Const_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_6"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_18/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_18"
  op: "Add"
  input: "training/Adam/Sqrt_6"
  input: "training/Adam/add_18/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_6"
  op: "RealDiv"
  input: "training/Adam/mul_30"
  input: "training/Adam/add_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_46"
  op: "ReadVariableOp"
  input: "conv2d_2/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_19"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_46"
  input: "training/Adam/truediv_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_15"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_5"
  input: "training/Adam/add_16"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_47"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_5"
  input: "^training/Adam/AssignVariableOp_15"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_16"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_21"
  input: "training/Adam/add_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_48"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_21"
  input: "^training/Adam/AssignVariableOp_16"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_17"
  op: "AssignVariableOp"
  input: "conv2d_2/bias"
  input: "training/Adam/sub_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_49"
  op: "ReadVariableOp"
  input: "conv2d_2/bias"
  input: "^training/Adam/AssignVariableOp_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_50"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_31/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_6"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_31"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_50"
  input: "training/Adam/mul_31/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_51"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_20/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_20"
  op: "Sub"
  input: "training/Adam/sub_20/x"
  input: "training/Adam/ReadVariableOp_51"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_32"
  op: "Mul"
  input: "training/Adam/sub_20"
  input: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_19"
  op: "Add"
  input: "training/Adam/mul_31"
  input: "training/Adam/mul_32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_52"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_33/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_33"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_52"
  input: "training/Adam/mul_33/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_53"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_21/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_21"
  op: "Sub"
  input: "training/Adam/sub_21/x"
  input: "training/Adam/ReadVariableOp_53"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_6"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_34"
  op: "Mul"
  input: "training/Adam/sub_21"
  input: "training/Adam/Square_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_20"
  op: "Add"
  input: "training/Adam/mul_33"
  input: "training/Adam/mul_34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_35"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_15"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_16"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_7/Minimum"
  op: "Minimum"
  input: "training/Adam/add_20"
  input: "training/Adam/Const_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_7"
  op: "Maximum"
  input: "training/Adam/clip_by_value_7/Minimum"
  input: "training/Adam/Const_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_7"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_21/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_21"
  op: "Add"
  input: "training/Adam/Sqrt_7"
  input: "training/Adam/add_21/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_7"
  op: "RealDiv"
  input: "training/Adam/mul_35"
  input: "training/Adam/add_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_54"
  op: "ReadVariableOp"
  input: "conv2d_transpose/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_22"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_54"
  input: "training/Adam/truediv_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_18"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_6"
  input: "training/Adam/add_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_55"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_6"
  input: "^training/Adam/AssignVariableOp_18"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_19"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_22"
  input: "training/Adam/add_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_56"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_22"
  input: "^training/Adam/AssignVariableOp_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_20"
  op: "AssignVariableOp"
  input: "conv2d_transpose/kernel"
  input: "training/Adam/sub_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_57"
  op: "ReadVariableOp"
  input: "conv2d_transpose/kernel"
  input: "^training/Adam/AssignVariableOp_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_58"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_36/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_36"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_58"
  input: "training/Adam/mul_36/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_59"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_23/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_23"
  op: "Sub"
  input: "training/Adam/sub_23/x"
  input: "training/Adam/ReadVariableOp_59"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_37"
  op: "Mul"
  input: "training/Adam/sub_23"
  input: "training/Adam/gradients/conv2d_transpose/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_22"
  op: "Add"
  input: "training/Adam/mul_36"
  input: "training/Adam/mul_37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_60"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_38/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_38"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_60"
  input: "training/Adam/mul_38/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_61"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_24/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_24"
  op: "Sub"
  input: "training/Adam/sub_24/x"
  input: "training/Adam/ReadVariableOp_61"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_7"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_39"
  op: "Mul"
  input: "training/Adam/sub_24"
  input: "training/Adam/Square_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_23"
  op: "Add"
  input: "training/Adam/mul_38"
  input: "training/Adam/mul_39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_40"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_17"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_18"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_8/Minimum"
  op: "Minimum"
  input: "training/Adam/add_23"
  input: "training/Adam/Const_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_8"
  op: "Maximum"
  input: "training/Adam/clip_by_value_8/Minimum"
  input: "training/Adam/Const_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_8"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_24/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_24"
  op: "Add"
  input: "training/Adam/Sqrt_8"
  input: "training/Adam/add_24/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_8"
  op: "RealDiv"
  input: "training/Adam/mul_40"
  input: "training/Adam/add_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_62"
  op: "ReadVariableOp"
  input: "conv2d_transpose/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_25"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_62"
  input: "training/Adam/truediv_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_21"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_7"
  input: "training/Adam/add_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_63"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_7"
  input: "^training/Adam/AssignVariableOp_21"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_22"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_23"
  input: "training/Adam/add_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_64"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_23"
  input: "^training/Adam/AssignVariableOp_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_23"
  op: "AssignVariableOp"
  input: "conv2d_transpose/bias"
  input: "training/Adam/sub_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_65"
  op: "ReadVariableOp"
  input: "conv2d_transpose/bias"
  input: "^training/Adam/AssignVariableOp_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_66"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_41/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_41"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_66"
  input: "training/Adam/mul_41/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_67"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_26/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_26"
  op: "Sub"
  input: "training/Adam/sub_26/x"
  input: "training/Adam/ReadVariableOp_67"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_42"
  op: "Mul"
  input: "training/Adam/sub_26"
  input: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_25"
  op: "Add"
  input: "training/Adam/mul_41"
  input: "training/Adam/mul_42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_68"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_43/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_24"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_43"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_68"
  input: "training/Adam/mul_43/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_69"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_27/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_27"
  op: "Sub"
  input: "training/Adam/sub_27/x"
  input: "training/Adam/ReadVariableOp_69"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_8"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_1/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_44"
  op: "Mul"
  input: "training/Adam/sub_27"
  input: "training/Adam/Square_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_26"
  op: "Add"
  input: "training/Adam/mul_43"
  input: "training/Adam/mul_44"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_45"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_19"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_20"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_9/Minimum"
  op: "Minimum"
  input: "training/Adam/add_26"
  input: "training/Adam/Const_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_9"
  op: "Maximum"
  input: "training/Adam/clip_by_value_9/Minimum"
  input: "training/Adam/Const_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_9"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_27/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_27"
  op: "Add"
  input: "training/Adam/Sqrt_9"
  input: "training/Adam/add_27/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_9"
  op: "RealDiv"
  input: "training/Adam/mul_45"
  input: "training/Adam/add_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_70"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_28"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_70"
  input: "training/Adam/truediv_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_24"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_8"
  input: "training/Adam/add_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_71"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_8"
  input: "^training/Adam/AssignVariableOp_24"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_25"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_24"
  input: "training/Adam/add_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_72"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_24"
  input: "^training/Adam/AssignVariableOp_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_26"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "training/Adam/sub_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_73"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "^training/Adam/AssignVariableOp_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_74"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_46/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_9"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_46"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_74"
  input: "training/Adam/mul_46/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_75"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_29/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_29"
  op: "Sub"
  input: "training/Adam/sub_29/x"
  input: "training/Adam/ReadVariableOp_75"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_47"
  op: "Mul"
  input: "training/Adam/sub_29"
  input: "training/Adam/gradients/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_28"
  op: "Add"
  input: "training/Adam/mul_46"
  input: "training/Adam/mul_47"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_76"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_48/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_48"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_76"
  input: "training/Adam/mul_48/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_77"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_30/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_30"
  op: "Sub"
  input: "training/Adam/sub_30/x"
  input: "training/Adam/ReadVariableOp_77"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_9"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_1/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_49"
  op: "Mul"
  input: "training/Adam/sub_30"
  input: "training/Adam/Square_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_29"
  op: "Add"
  input: "training/Adam/mul_48"
  input: "training/Adam/mul_49"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_50"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_21"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_22"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_10/Minimum"
  op: "Minimum"
  input: "training/Adam/add_29"
  input: "training/Adam/Const_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_10"
  op: "Maximum"
  input: "training/Adam/clip_by_value_10/Minimum"
  input: "training/Adam/Const_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_10"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_30/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_30"
  op: "Add"
  input: "training/Adam/Sqrt_10"
  input: "training/Adam/add_30/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_10"
  op: "RealDiv"
  input: "training/Adam/mul_50"
  input: "training/Adam/add_30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_78"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_31"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_78"
  input: "training/Adam/truediv_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_27"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_9"
  input: "training/Adam/add_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_79"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_9"
  input: "^training/Adam/AssignVariableOp_27"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_28"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_25"
  input: "training/Adam/add_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_80"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_25"
  input: "^training/Adam/AssignVariableOp_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_29"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "training/Adam/sub_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_81"
  op: "ReadVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "^training/Adam/AssignVariableOp_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_82"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_51/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_51"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_82"
  input: "training/Adam/mul_51/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_83"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_32/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_32"
  op: "Sub"
  input: "training/Adam/sub_32/x"
  input: "training/Adam/ReadVariableOp_83"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_52"
  op: "Mul"
  input: "training/Adam/sub_32"
  input: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_31"
  op: "Add"
  input: "training/Adam/mul_51"
  input: "training/Adam/mul_52"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_84"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_53/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_53"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_84"
  input: "training/Adam/mul_53/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_85"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_33/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_33"
  op: "Sub"
  input: "training/Adam/sub_33/x"
  input: "training/Adam/ReadVariableOp_85"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_10"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_2/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_54"
  op: "Mul"
  input: "training/Adam/sub_33"
  input: "training/Adam/Square_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_32"
  op: "Add"
  input: "training/Adam/mul_53"
  input: "training/Adam/mul_54"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_55"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_23"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_24"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_11/Minimum"
  op: "Minimum"
  input: "training/Adam/add_32"
  input: "training/Adam/Const_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_11"
  op: "Maximum"
  input: "training/Adam/clip_by_value_11/Minimum"
  input: "training/Adam/Const_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_11"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_33/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_33"
  op: "Add"
  input: "training/Adam/Sqrt_11"
  input: "training/Adam/add_33/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_11"
  op: "RealDiv"
  input: "training/Adam/mul_55"
  input: "training/Adam/add_33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_86"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_34"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_86"
  input: "training/Adam/truediv_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_30"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_10"
  input: "training/Adam/add_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_87"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_10"
  input: "^training/Adam/AssignVariableOp_30"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_31"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_26"
  input: "training/Adam/add_32"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_88"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_26"
  input: "^training/Adam/AssignVariableOp_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_32"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "training/Adam/sub_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_89"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "^training/Adam/AssignVariableOp_32"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_90"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_56/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_56"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_90"
  input: "training/Adam/mul_56/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_91"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_35/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_35"
  op: "Sub"
  input: "training/Adam/sub_35/x"
  input: "training/Adam/ReadVariableOp_91"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_57"
  op: "Mul"
  input: "training/Adam/sub_35"
  input: "training/Adam/gradients/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_34"
  op: "Add"
  input: "training/Adam/mul_56"
  input: "training/Adam/mul_57"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_92"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_58/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_27"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_58"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_92"
  input: "training/Adam/mul_58/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_93"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_36/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_36"
  op: "Sub"
  input: "training/Adam/sub_36/x"
  input: "training/Adam/ReadVariableOp_93"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_11"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_2/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_59"
  op: "Mul"
  input: "training/Adam/sub_36"
  input: "training/Adam/Square_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_35"
  op: "Add"
  input: "training/Adam/mul_58"
  input: "training/Adam/mul_59"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_60"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_25"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_26"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_12/Minimum"
  op: "Minimum"
  input: "training/Adam/add_35"
  input: "training/Adam/Const_26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_12"
  op: "Maximum"
  input: "training/Adam/clip_by_value_12/Minimum"
  input: "training/Adam/Const_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_12"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_36/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_36"
  op: "Add"
  input: "training/Adam/Sqrt_12"
  input: "training/Adam/add_36/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_12"
  op: "RealDiv"
  input: "training/Adam/mul_60"
  input: "training/Adam/add_36"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_94"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_37"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_94"
  input: "training/Adam/truediv_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_33"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_11"
  input: "training/Adam/add_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_95"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_11"
  input: "^training/Adam/AssignVariableOp_33"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_34"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_27"
  input: "training/Adam/add_35"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_96"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_27"
  input: "^training/Adam/AssignVariableOp_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_35"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "training/Adam/sub_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_97"
  op: "ReadVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "^training/Adam/AssignVariableOp_35"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_98"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_61/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_12"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_61"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_98"
  input: "training/Adam/mul_61/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_99"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_38/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_38"
  op: "Sub"
  input: "training/Adam/sub_38/x"
  input: "training/Adam/ReadVariableOp_99"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_62"
  op: "Mul"
  input: "training/Adam/sub_38"
  input: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_37"
  op: "Add"
  input: "training/Adam/mul_61"
  input: "training/Adam/mul_62"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_100"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_63/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_63"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_100"
  input: "training/Adam/mul_63/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_101"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_39/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_39"
  op: "Sub"
  input: "training/Adam/sub_39/x"
  input: "training/Adam/ReadVariableOp_101"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_12"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_3/conv2d_transpose_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_64"
  op: "Mul"
  input: "training/Adam/sub_39"
  input: "training/Adam/Square_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_38"
  op: "Add"
  input: "training/Adam/mul_63"
  input: "training/Adam/mul_64"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_65"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_27"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_28"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_13/Minimum"
  op: "Minimum"
  input: "training/Adam/add_38"
  input: "training/Adam/Const_28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_13"
  op: "Maximum"
  input: "training/Adam/clip_by_value_13/Minimum"
  input: "training/Adam/Const_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_13"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_39/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_39"
  op: "Add"
  input: "training/Adam/Sqrt_13"
  input: "training/Adam/add_39/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_13"
  op: "RealDiv"
  input: "training/Adam/mul_65"
  input: "training/Adam/add_39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_102"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_40"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_102"
  input: "training/Adam/truediv_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_36"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_12"
  input: "training/Adam/add_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_103"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_12"
  input: "^training/Adam/AssignVariableOp_36"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_37"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_28"
  input: "training/Adam/add_38"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_104"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_28"
  input: "^training/Adam/AssignVariableOp_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_38"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "training/Adam/sub_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_105"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "^training/Adam/AssignVariableOp_38"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_106"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_66/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_66"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_106"
  input: "training/Adam/mul_66/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_107"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_41/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_41"
  op: "Sub"
  input: "training/Adam/sub_41/x"
  input: "training/Adam/ReadVariableOp_107"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_67"
  op: "Mul"
  input: "training/Adam/sub_41"
  input: "training/Adam/gradients/conv2d_transpose_3/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_40"
  op: "Add"
  input: "training/Adam/mul_66"
  input: "training/Adam/mul_67"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_108"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_68/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_68"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_108"
  input: "training/Adam/mul_68/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_109"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_42/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_42"
  op: "Sub"
  input: "training/Adam/sub_42/x"
  input: "training/Adam/ReadVariableOp_109"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_13"
  op: "Square"
  input: "training/Adam/gradients/conv2d_transpose_3/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_69"
  op: "Mul"
  input: "training/Adam/sub_42"
  input: "training/Adam/Square_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_41"
  op: "Add"
  input: "training/Adam/mul_68"
  input: "training/Adam/mul_69"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_70"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_40"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_29"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_30"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_14/Minimum"
  op: "Minimum"
  input: "training/Adam/add_41"
  input: "training/Adam/Const_30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_14"
  op: "Maximum"
  input: "training/Adam/clip_by_value_14/Minimum"
  input: "training/Adam/Const_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_14"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_42/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_42"
  op: "Add"
  input: "training/Adam/Sqrt_14"
  input: "training/Adam/add_42/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_14"
  op: "RealDiv"
  input: "training/Adam/mul_70"
  input: "training/Adam/add_42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_110"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_43"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_110"
  input: "training/Adam/truediv_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_39"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_13"
  input: "training/Adam/add_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_111"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_13"
  input: "^training/Adam/AssignVariableOp_39"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_40"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_29"
  input: "training/Adam/add_41"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_112"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_29"
  input: "^training/Adam/AssignVariableOp_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_41"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "training/Adam/sub_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_113"
  op: "ReadVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "^training/Adam/AssignVariableOp_41"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_114"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_71/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_71"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_114"
  input: "training/Adam/mul_71/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_115"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_44/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_44"
  op: "Sub"
  input: "training/Adam/sub_44/x"
  input: "training/Adam/ReadVariableOp_115"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_72"
  op: "Mul"
  input: "training/Adam/sub_44"
  input: "training/Adam/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_43"
  op: "Add"
  input: "training/Adam/mul_71"
  input: "training/Adam/mul_72"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_116"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_73/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_30"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_73"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_116"
  input: "training/Adam/mul_73/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_117"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_45/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_45"
  op: "Sub"
  input: "training/Adam/sub_45/x"
  input: "training/Adam/ReadVariableOp_117"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_14"
  op: "Square"
  input: "training/Adam/gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_74"
  op: "Mul"
  input: "training/Adam/sub_45"
  input: "training/Adam/Square_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_44"
  op: "Add"
  input: "training/Adam/mul_73"
  input: "training/Adam/mul_74"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_75"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_31"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_32"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_15/Minimum"
  op: "Minimum"
  input: "training/Adam/add_44"
  input: "training/Adam/Const_32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_15"
  op: "Maximum"
  input: "training/Adam/clip_by_value_15/Minimum"
  input: "training/Adam/Const_31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_15"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_45/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_45"
  op: "Add"
  input: "training/Adam/Sqrt_15"
  input: "training/Adam/add_45/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_15"
  op: "RealDiv"
  input: "training/Adam/mul_75"
  input: "training/Adam/add_45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_118"
  op: "ReadVariableOp"
  input: "conv2d_3/kernel"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_46"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_118"
  input: "training/Adam/truediv_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_42"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_14"
  input: "training/Adam/add_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_119"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_14"
  input: "^training/Adam/AssignVariableOp_42"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_43"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_30"
  input: "training/Adam/add_44"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_120"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_30"
  input: "^training/Adam/AssignVariableOp_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_44"
  op: "AssignVariableOp"
  input: "conv2d_3/kernel"
  input: "training/Adam/sub_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_121"
  op: "ReadVariableOp"
  input: "conv2d_3/kernel"
  input: "^training/Adam/AssignVariableOp_44"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_122"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_76/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_15"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_76"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_122"
  input: "training/Adam/mul_76/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_123"
  op: "ReadVariableOp"
  input: "Adam/beta_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_47/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_47"
  op: "Sub"
  input: "training/Adam/sub_47/x"
  input: "training/Adam/ReadVariableOp_123"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_77"
  op: "Mul"
  input: "training/Adam/sub_47"
  input: "training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_46"
  op: "Add"
  input: "training/Adam/mul_76"
  input: "training/Adam/mul_77"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_124"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_78/ReadVariableOp"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_78"
  op: "Mul"
  input: "training/Adam/ReadVariableOp_124"
  input: "training/Adam/mul_78/ReadVariableOp"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_125"
  op: "ReadVariableOp"
  input: "Adam/beta_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_48/x"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "training/Adam/sub_48"
  op: "Sub"
  input: "training/Adam/sub_48/x"
  input: "training/Adam/ReadVariableOp_125"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Square_15"
  op: "Square"
  input: "training/Adam/gradients/conv2d_3/BiasAdd_grad/BiasAddGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_79"
  op: "Mul"
  input: "training/Adam/sub_48"
  input: "training/Adam/Square_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_47"
  op: "Add"
  input: "training/Adam/mul_78"
  input: "training/Adam/mul_79"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/mul_80"
  op: "Mul"
  input: "training/Adam/mul"
  input: "training/Adam/add_46"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Const_33"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "training/Adam/Const_34"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: inf
      }
    }
  }
}
node {
  name: "training/Adam/clip_by_value_16/Minimum"
  op: "Minimum"
  input: "training/Adam/add_47"
  input: "training/Adam/Const_34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/clip_by_value_16"
  op: "Maximum"
  input: "training/Adam/clip_by_value_16/Minimum"
  input: "training/Adam/Const_33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/Sqrt_16"
  op: "Sqrt"
  input: "training/Adam/clip_by_value_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/add_48/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0000000116860974e-07
      }
    }
  }
}
node {
  name: "training/Adam/add_48"
  op: "Add"
  input: "training/Adam/Sqrt_16"
  input: "training/Adam/add_48/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/truediv_16"
  op: "RealDiv"
  input: "training/Adam/mul_80"
  input: "training/Adam/add_48"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_126"
  op: "ReadVariableOp"
  input: "conv2d_3/bias"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/sub_49"
  op: "Sub"
  input: "training/Adam/ReadVariableOp_126"
  input: "training/Adam/truediv_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_45"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_15"
  input: "training/Adam/add_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_127"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_15"
  input: "^training/Adam/AssignVariableOp_45"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_46"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_31"
  input: "training/Adam/add_47"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_128"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_31"
  input: "^training/Adam/AssignVariableOp_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/AssignVariableOp_47"
  op: "AssignVariableOp"
  input: "conv2d_3/bias"
  input: "training/Adam/sub_49"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/Adam/ReadVariableOp_129"
  op: "ReadVariableOp"
  input: "conv2d_3/bias"
  input: "^training/Adam/AssignVariableOp_47"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "training/group_deps"
  op: "NoOp"
  input: "^loss/mul"
  input: "^metrics/mean_absolute_error/Mean_1"
  input: "^training/Adam/ReadVariableOp"
  input: "^training/Adam/ReadVariableOp_103"
  input: "^training/Adam/ReadVariableOp_104"
  input: "^training/Adam/ReadVariableOp_105"
  input: "^training/Adam/ReadVariableOp_111"
  input: "^training/Adam/ReadVariableOp_112"
  input: "^training/Adam/ReadVariableOp_113"
  input: "^training/Adam/ReadVariableOp_119"
  input: "^training/Adam/ReadVariableOp_120"
  input: "^training/Adam/ReadVariableOp_121"
  input: "^training/Adam/ReadVariableOp_127"
  input: "^training/Adam/ReadVariableOp_128"
  input: "^training/Adam/ReadVariableOp_129"
  input: "^training/Adam/ReadVariableOp_15"
  input: "^training/Adam/ReadVariableOp_16"
  input: "^training/Adam/ReadVariableOp_17"
  input: "^training/Adam/ReadVariableOp_23"
  input: "^training/Adam/ReadVariableOp_24"
  input: "^training/Adam/ReadVariableOp_25"
  input: "^training/Adam/ReadVariableOp_31"
  input: "^training/Adam/ReadVariableOp_32"
  input: "^training/Adam/ReadVariableOp_33"
  input: "^training/Adam/ReadVariableOp_39"
  input: "^training/Adam/ReadVariableOp_40"
  input: "^training/Adam/ReadVariableOp_41"
  input: "^training/Adam/ReadVariableOp_47"
  input: "^training/Adam/ReadVariableOp_48"
  input: "^training/Adam/ReadVariableOp_49"
  input: "^training/Adam/ReadVariableOp_55"
  input: "^training/Adam/ReadVariableOp_56"
  input: "^training/Adam/ReadVariableOp_57"
  input: "^training/Adam/ReadVariableOp_63"
  input: "^training/Adam/ReadVariableOp_64"
  input: "^training/Adam/ReadVariableOp_65"
  input: "^training/Adam/ReadVariableOp_7"
  input: "^training/Adam/ReadVariableOp_71"
  input: "^training/Adam/ReadVariableOp_72"
  input: "^training/Adam/ReadVariableOp_73"
  input: "^training/Adam/ReadVariableOp_79"
  input: "^training/Adam/ReadVariableOp_8"
  input: "^training/Adam/ReadVariableOp_80"
  input: "^training/Adam/ReadVariableOp_81"
  input: "^training/Adam/ReadVariableOp_87"
  input: "^training/Adam/ReadVariableOp_88"
  input: "^training/Adam/ReadVariableOp_89"
  input: "^training/Adam/ReadVariableOp_9"
  input: "^training/Adam/ReadVariableOp_95"
  input: "^training/Adam/ReadVariableOp_96"
  input: "^training/Adam/ReadVariableOp_97"
}
node {
  name: "VarIsInitializedOp_16"
  op: "VarIsInitializedOp"
  input: "Adam/iterations"
}
node {
  name: "VarIsInitializedOp_17"
  op: "VarIsInitializedOp"
  input: "Adam/lr"
}
node {
  name: "VarIsInitializedOp_18"
  op: "VarIsInitializedOp"
  input: "Adam/beta_1"
}
node {
  name: "VarIsInitializedOp_19"
  op: "VarIsInitializedOp"
  input: "Adam/beta_2"
}
node {
  name: "VarIsInitializedOp_20"
  op: "VarIsInitializedOp"
  input: "Adam/decay"
}
node {
  name: "VarIsInitializedOp_21"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable"
}
node {
  name: "VarIsInitializedOp_22"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_1"
}
node {
  name: "VarIsInitializedOp_23"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_2"
}
node {
  name: "VarIsInitializedOp_24"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_3"
}
node {
  name: "VarIsInitializedOp_25"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_4"
}
node {
  name: "VarIsInitializedOp_26"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_5"
}
node {
  name: "VarIsInitializedOp_27"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_6"
}
node {
  name: "VarIsInitializedOp_28"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_7"
}
node {
  name: "VarIsInitializedOp_29"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_8"
}
node {
  name: "VarIsInitializedOp_30"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_9"
}
node {
  name: "VarIsInitializedOp_31"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_10"
}
node {
  name: "VarIsInitializedOp_32"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_11"
}
node {
  name: "VarIsInitializedOp_33"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_12"
}
node {
  name: "VarIsInitializedOp_34"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_13"
}
node {
  name: "VarIsInitializedOp_35"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_14"
}
node {
  name: "VarIsInitializedOp_36"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_15"
}
node {
  name: "VarIsInitializedOp_37"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_16"
}
node {
  name: "VarIsInitializedOp_38"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_17"
}
node {
  name: "VarIsInitializedOp_39"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_18"
}
node {
  name: "VarIsInitializedOp_40"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_19"
}
node {
  name: "VarIsInitializedOp_41"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_20"
}
node {
  name: "VarIsInitializedOp_42"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_21"
}
node {
  name: "VarIsInitializedOp_43"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_22"
}
node {
  name: "VarIsInitializedOp_44"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_23"
}
node {
  name: "VarIsInitializedOp_45"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_24"
}
node {
  name: "VarIsInitializedOp_46"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_25"
}
node {
  name: "VarIsInitializedOp_47"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_26"
}
node {
  name: "VarIsInitializedOp_48"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_27"
}
node {
  name: "VarIsInitializedOp_49"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_28"
}
node {
  name: "VarIsInitializedOp_50"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_29"
}
node {
  name: "VarIsInitializedOp_51"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_30"
}
node {
  name: "VarIsInitializedOp_52"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_31"
}
node {
  name: "VarIsInitializedOp_53"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_32"
}
node {
  name: "VarIsInitializedOp_54"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_33"
}
node {
  name: "VarIsInitializedOp_55"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_34"
}
node {
  name: "VarIsInitializedOp_56"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_35"
}
node {
  name: "VarIsInitializedOp_57"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_36"
}
node {
  name: "VarIsInitializedOp_58"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_37"
}
node {
  name: "VarIsInitializedOp_59"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_38"
}
node {
  name: "VarIsInitializedOp_60"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_39"
}
node {
  name: "VarIsInitializedOp_61"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_40"
}
node {
  name: "VarIsInitializedOp_62"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_41"
}
node {
  name: "VarIsInitializedOp_63"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_42"
}
node {
  name: "VarIsInitializedOp_64"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_43"
}
node {
  name: "VarIsInitializedOp_65"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_44"
}
node {
  name: "VarIsInitializedOp_66"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_45"
}
node {
  name: "VarIsInitializedOp_67"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_46"
}
node {
  name: "VarIsInitializedOp_68"
  op: "VarIsInitializedOp"
  input: "training/Adam/Variable_47"
}
node {
  name: "init_1"
  op: "NoOp"
  input: "^Adam/beta_1/Assign"
  input: "^Adam/beta_2/Assign"
  input: "^Adam/decay/Assign"
  input: "^Adam/iterations/Assign"
  input: "^Adam/lr/Assign"
  input: "^training/Adam/Variable/Assign"
  input: "^training/Adam/Variable_1/Assign"
  input: "^training/Adam/Variable_10/Assign"
  input: "^training/Adam/Variable_11/Assign"
  input: "^training/Adam/Variable_12/Assign"
  input: "^training/Adam/Variable_13/Assign"
  input: "^training/Adam/Variable_14/Assign"
  input: "^training/Adam/Variable_15/Assign"
  input: "^training/Adam/Variable_16/Assign"
  input: "^training/Adam/Variable_17/Assign"
  input: "^training/Adam/Variable_18/Assign"
  input: "^training/Adam/Variable_19/Assign"
  input: "^training/Adam/Variable_2/Assign"
  input: "^training/Adam/Variable_20/Assign"
  input: "^training/Adam/Variable_21/Assign"
  input: "^training/Adam/Variable_22/Assign"
  input: "^training/Adam/Variable_23/Assign"
  input: "^training/Adam/Variable_24/Assign"
  input: "^training/Adam/Variable_25/Assign"
  input: "^training/Adam/Variable_26/Assign"
  input: "^training/Adam/Variable_27/Assign"
  input: "^training/Adam/Variable_28/Assign"
  input: "^training/Adam/Variable_29/Assign"
  input: "^training/Adam/Variable_3/Assign"
  input: "^training/Adam/Variable_30/Assign"
  input: "^training/Adam/Variable_31/Assign"
  input: "^training/Adam/Variable_32/Assign"
  input: "^training/Adam/Variable_33/Assign"
  input: "^training/Adam/Variable_34/Assign"
  input: "^training/Adam/Variable_35/Assign"
  input: "^training/Adam/Variable_36/Assign"
  input: "^training/Adam/Variable_37/Assign"
  input: "^training/Adam/Variable_38/Assign"
  input: "^training/Adam/Variable_39/Assign"
  input: "^training/Adam/Variable_4/Assign"
  input: "^training/Adam/Variable_40/Assign"
  input: "^training/Adam/Variable_41/Assign"
  input: "^training/Adam/Variable_42/Assign"
  input: "^training/Adam/Variable_43/Assign"
  input: "^training/Adam/Variable_44/Assign"
  input: "^training/Adam/Variable_45/Assign"
  input: "^training/Adam/Variable_46/Assign"
  input: "^training/Adam/Variable_47/Assign"
  input: "^training/Adam/Variable_5/Assign"
  input: "^training/Adam/Variable_6/Assign"
  input: "^training/Adam/Variable_7/Assign"
  input: "^training/Adam/Variable_8/Assign"
  input: "^training/Adam/Variable_9/Assign"
}
node {
  name: "Placeholder_16"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "AssignVariableOp_16"
  op: "AssignVariableOp"
  input: "Adam/iterations"
  input: "Placeholder_16"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "ReadVariableOp_16"
  op: "ReadVariableOp"
  input: "Adam/iterations"
  input: "^AssignVariableOp_16"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Placeholder_17"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_17"
  op: "AssignVariableOp"
  input: "training/Adam/Variable"
  input: "Placeholder_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_17"
  op: "ReadVariableOp"
  input: "training/Adam/Variable"
  input: "^AssignVariableOp_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_18"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_18"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_1"
  input: "Placeholder_18"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_18"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_1"
  input: "^AssignVariableOp_18"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_19"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_19"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_2"
  input: "Placeholder_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_19"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_2"
  input: "^AssignVariableOp_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_20"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_20"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_3"
  input: "Placeholder_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_20"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_3"
  input: "^AssignVariableOp_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_21"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_21"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_4"
  input: "Placeholder_21"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_21"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_4"
  input: "^AssignVariableOp_21"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_22"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_22"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_5"
  input: "Placeholder_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_22"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_5"
  input: "^AssignVariableOp_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_23"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_23"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_6"
  input: "Placeholder_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_23"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_6"
  input: "^AssignVariableOp_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_24"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_24"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_7"
  input: "Placeholder_24"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_24"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_7"
  input: "^AssignVariableOp_24"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_25"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_25"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_8"
  input: "Placeholder_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_25"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_8"
  input: "^AssignVariableOp_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_26"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_26"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_9"
  input: "Placeholder_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_26"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_9"
  input: "^AssignVariableOp_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_27"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_27"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_10"
  input: "Placeholder_27"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_27"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_10"
  input: "^AssignVariableOp_27"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_28"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_28"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_11"
  input: "Placeholder_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_28"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_11"
  input: "^AssignVariableOp_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_29"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_29"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_12"
  input: "Placeholder_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_29"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_12"
  input: "^AssignVariableOp_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_30"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_30"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_13"
  input: "Placeholder_30"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_30"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_13"
  input: "^AssignVariableOp_30"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_31"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_31"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_14"
  input: "Placeholder_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_31"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_14"
  input: "^AssignVariableOp_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_32"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_32"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_15"
  input: "Placeholder_32"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_32"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_15"
  input: "^AssignVariableOp_32"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_33"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 7
        }
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_33"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_16"
  input: "Placeholder_33"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_33"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_16"
  input: "^AssignVariableOp_33"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_34"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_34"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_17"
  input: "Placeholder_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_34"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_17"
  input: "^AssignVariableOp_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_35"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_35"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_18"
  input: "Placeholder_35"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_35"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_18"
  input: "^AssignVariableOp_35"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_36"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_36"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_19"
  input: "Placeholder_36"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_36"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_19"
  input: "^AssignVariableOp_36"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_37"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_37"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_20"
  input: "Placeholder_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_37"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_20"
  input: "^AssignVariableOp_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_38"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_38"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_21"
  input: "Placeholder_38"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_38"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_21"
  input: "^AssignVariableOp_38"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_39"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 128
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_39"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_22"
  input: "Placeholder_39"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_39"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_22"
  input: "^AssignVariableOp_39"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_40"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_40"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_23"
  input: "Placeholder_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_40"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_23"
  input: "^AssignVariableOp_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_41"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 64
        }
        dim {
          size: 128
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_41"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_24"
  input: "Placeholder_41"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_41"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_24"
  input: "^AssignVariableOp_41"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_42"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_42"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_25"
  input: "Placeholder_42"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_42"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_25"
  input: "^AssignVariableOp_42"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_43"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_43"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_26"
  input: "Placeholder_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_43"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_26"
  input: "^AssignVariableOp_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_44"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_44"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_27"
  input: "Placeholder_44"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_44"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_27"
  input: "^AssignVariableOp_44"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_45"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 32
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_45"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_28"
  input: "Placeholder_45"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_45"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_28"
  input: "^AssignVariableOp_45"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_46"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_46"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_29"
  input: "Placeholder_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_46"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_29"
  input: "^AssignVariableOp_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_47"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 16
        }
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_47"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_30"
  input: "Placeholder_47"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_47"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_30"
  input: "^AssignVariableOp_47"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_48"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 7
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_48"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_31"
  input: "Placeholder_48"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_48"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_31"
  input: "^AssignVariableOp_48"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_49"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_49"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_32"
  input: "Placeholder_49"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_49"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_32"
  input: "^AssignVariableOp_49"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_50"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_50"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_33"
  input: "Placeholder_50"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_50"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_33"
  input: "^AssignVariableOp_50"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_51"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_51"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_34"
  input: "Placeholder_51"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_51"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_34"
  input: "^AssignVariableOp_51"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_52"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_52"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_35"
  input: "Placeholder_52"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_52"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_35"
  input: "^AssignVariableOp_52"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_53"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_53"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_36"
  input: "Placeholder_53"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_53"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_36"
  input: "^AssignVariableOp_53"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_54"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_54"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_37"
  input: "Placeholder_54"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_54"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_37"
  input: "^AssignVariableOp_54"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_55"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_55"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_38"
  input: "Placeholder_55"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_55"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_38"
  input: "^AssignVariableOp_55"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_56"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_56"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_39"
  input: "Placeholder_56"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_56"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_39"
  input: "^AssignVariableOp_56"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_57"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_57"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_40"
  input: "Placeholder_57"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_57"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_40"
  input: "^AssignVariableOp_57"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_58"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_58"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_41"
  input: "Placeholder_58"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_58"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_41"
  input: "^AssignVariableOp_58"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_59"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_59"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_42"
  input: "Placeholder_59"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_59"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_42"
  input: "^AssignVariableOp_59"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_60"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_60"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_43"
  input: "Placeholder_60"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_60"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_43"
  input: "^AssignVariableOp_60"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_61"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_61"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_44"
  input: "Placeholder_61"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_61"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_44"
  input: "^AssignVariableOp_61"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_62"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_62"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_45"
  input: "Placeholder_62"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_62"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_45"
  input: "^AssignVariableOp_62"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_63"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_63"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_46"
  input: "Placeholder_63"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_63"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_46"
  input: "^AssignVariableOp_63"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Placeholder_64"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
      }
    }
  }
}
node {
  name: "AssignVariableOp_64"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_47"
  input: "Placeholder_64"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ReadVariableOp_64"
  op: "ReadVariableOp"
  input: "training/Adam/Variable_47"
  input: "^AssignVariableOp_64"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 69
          }
        }
        string_val: "Adam/beta_1"
        string_val: "Adam/beta_2"
        string_val: "Adam/decay"
        string_val: "Adam/iterations"
        string_val: "Adam/lr"
        string_val: "conv2d/bias"
        string_val: "conv2d/kernel"
        string_val: "conv2d_1/bias"
        string_val: "conv2d_1/kernel"
        string_val: "conv2d_2/bias"
        string_val: "conv2d_2/kernel"
        string_val: "conv2d_3/bias"
        string_val: "conv2d_3/kernel"
        string_val: "conv2d_transpose/bias"
        string_val: "conv2d_transpose/kernel"
        string_val: "conv2d_transpose_1/bias"
        string_val: "conv2d_transpose_1/kernel"
        string_val: "conv2d_transpose_2/bias"
        string_val: "conv2d_transpose_2/kernel"
        string_val: "conv2d_transpose_3/bias"
        string_val: "conv2d_transpose_3/kernel"
        string_val: "training/Adam/Variable"
        string_val: "training/Adam/Variable_1"
        string_val: "training/Adam/Variable_10"
        string_val: "training/Adam/Variable_11"
        string_val: "training/Adam/Variable_12"
        string_val: "training/Adam/Variable_13"
        string_val: "training/Adam/Variable_14"
        string_val: "training/Adam/Variable_15"
        string_val: "training/Adam/Variable_16"
        string_val: "training/Adam/Variable_17"
        string_val: "training/Adam/Variable_18"
        string_val: "training/Adam/Variable_19"
        string_val: "training/Adam/Variable_2"
        string_val: "training/Adam/Variable_20"
        string_val: "training/Adam/Variable_21"
        string_val: "training/Adam/Variable_22"
        string_val: "training/Adam/Variable_23"
        string_val: "training/Adam/Variable_24"
        string_val: "training/Adam/Variable_25"
        string_val: "training/Adam/Variable_26"
        string_val: "training/Adam/Variable_27"
        string_val: "training/Adam/Variable_28"
        string_val: "training/Adam/Variable_29"
        string_val: "training/Adam/Variable_3"
        string_val: "training/Adam/Variable_30"
        string_val: "training/Adam/Variable_31"
        string_val: "training/Adam/Variable_32"
        string_val: "training/Adam/Variable_33"
        string_val: "training/Adam/Variable_34"
        string_val: "training/Adam/Variable_35"
        string_val: "training/Adam/Variable_36"
        string_val: "training/Adam/Variable_37"
        string_val: "training/Adam/Variable_38"
        string_val: "training/Adam/Variable_39"
        string_val: "training/Adam/Variable_4"
        string_val: "training/Adam/Variable_40"
        string_val: "training/Adam/Variable_41"
        string_val: "training/Adam/Variable_42"
        string_val: "training/Adam/Variable_43"
        string_val: "training/Adam/Variable_44"
        string_val: "training/Adam/Variable_45"
        string_val: "training/Adam/Variable_46"
        string_val: "training/Adam/Variable_47"
        string_val: "training/Adam/Variable_5"
        string_val: "training/Adam/Variable_6"
        string_val: "training/Adam/Variable_7"
        string_val: "training/Adam/Variable_8"
        string_val: "training/Adam/Variable_9"
      }
    }
  }
}
node {
  name: "save/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 69
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/SaveV2"
  op: "SaveV2"
  input: "save/Const"
  input: "save/SaveV2/tensor_names"
  input: "save/SaveV2/shape_and_slices"
  input: "Adam/beta_1/Read/ReadVariableOp"
  input: "Adam/beta_2/Read/ReadVariableOp"
  input: "Adam/decay/Read/ReadVariableOp"
  input: "Adam/iterations/Read/ReadVariableOp"
  input: "Adam/lr/Read/ReadVariableOp"
  input: "conv2d/bias/Read/ReadVariableOp"
  input: "conv2d/kernel/Read/ReadVariableOp"
  input: "conv2d_1/bias/Read/ReadVariableOp"
  input: "conv2d_1/kernel/Read/ReadVariableOp"
  input: "conv2d_2/bias/Read/ReadVariableOp"
  input: "conv2d_2/kernel/Read/ReadVariableOp"
  input: "conv2d_3/bias/Read/ReadVariableOp"
  input: "conv2d_3/kernel/Read/ReadVariableOp"
  input: "conv2d_transpose/bias/Read/ReadVariableOp"
  input: "conv2d_transpose/kernel/Read/ReadVariableOp"
  input: "conv2d_transpose_1/bias/Read/ReadVariableOp"
  input: "conv2d_transpose_1/kernel/Read/ReadVariableOp"
  input: "conv2d_transpose_2/bias/Read/ReadVariableOp"
  input: "conv2d_transpose_2/kernel/Read/ReadVariableOp"
  input: "conv2d_transpose_3/bias/Read/ReadVariableOp"
  input: "conv2d_transpose_3/kernel/Read/ReadVariableOp"
  input: "training/Adam/Variable/Read/ReadVariableOp"
  input: "training/Adam/Variable_1/Read/ReadVariableOp"
  input: "training/Adam/Variable_10/Read/ReadVariableOp"
  input: "training/Adam/Variable_11/Read/ReadVariableOp"
  input: "training/Adam/Variable_12/Read/ReadVariableOp"
  input: "training/Adam/Variable_13/Read/ReadVariableOp"
  input: "training/Adam/Variable_14/Read/ReadVariableOp"
  input: "training/Adam/Variable_15/Read/ReadVariableOp"
  input: "training/Adam/Variable_16/Read/ReadVariableOp"
  input: "training/Adam/Variable_17/Read/ReadVariableOp"
  input: "training/Adam/Variable_18/Read/ReadVariableOp"
  input: "training/Adam/Variable_19/Read/ReadVariableOp"
  input: "training/Adam/Variable_2/Read/ReadVariableOp"
  input: "training/Adam/Variable_20/Read/ReadVariableOp"
  input: "training/Adam/Variable_21/Read/ReadVariableOp"
  input: "training/Adam/Variable_22/Read/ReadVariableOp"
  input: "training/Adam/Variable_23/Read/ReadVariableOp"
  input: "training/Adam/Variable_24/Read/ReadVariableOp"
  input: "training/Adam/Variable_25/Read/ReadVariableOp"
  input: "training/Adam/Variable_26/Read/ReadVariableOp"
  input: "training/Adam/Variable_27/Read/ReadVariableOp"
  input: "training/Adam/Variable_28/Read/ReadVariableOp"
  input: "training/Adam/Variable_29/Read/ReadVariableOp"
  input: "training/Adam/Variable_3/Read/ReadVariableOp"
  input: "training/Adam/Variable_30/Read/ReadVariableOp"
  input: "training/Adam/Variable_31/Read/ReadVariableOp"
  input: "training/Adam/Variable_32/Read/ReadVariableOp"
  input: "training/Adam/Variable_33/Read/ReadVariableOp"
  input: "training/Adam/Variable_34/Read/ReadVariableOp"
  input: "training/Adam/Variable_35/Read/ReadVariableOp"
  input: "training/Adam/Variable_36/Read/ReadVariableOp"
  input: "training/Adam/Variable_37/Read/ReadVariableOp"
  input: "training/Adam/Variable_38/Read/ReadVariableOp"
  input: "training/Adam/Variable_39/Read/ReadVariableOp"
  input: "training/Adam/Variable_4/Read/ReadVariableOp"
  input: "training/Adam/Variable_40/Read/ReadVariableOp"
  input: "training/Adam/Variable_41/Read/ReadVariableOp"
  input: "training/Adam/Variable_42/Read/ReadVariableOp"
  input: "training/Adam/Variable_43/Read/ReadVariableOp"
  input: "training/Adam/Variable_44/Read/ReadVariableOp"
  input: "training/Adam/Variable_45/Read/ReadVariableOp"
  input: "training/Adam/Variable_46/Read/ReadVariableOp"
  input: "training/Adam/Variable_47/Read/ReadVariableOp"
  input: "training/Adam/Variable_5/Read/ReadVariableOp"
  input: "training/Adam/Variable_6/Read/ReadVariableOp"
  input: "training/Adam/Variable_7/Read/ReadVariableOp"
  input: "training/Adam/Variable_8/Read/ReadVariableOp"
  input: "training/Adam/Variable_9/Read/ReadVariableOp"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_INT64
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  device: "/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 69
          }
        }
        string_val: "Adam/beta_1"
        string_val: "Adam/beta_2"
        string_val: "Adam/decay"
        string_val: "Adam/iterations"
        string_val: "Adam/lr"
        string_val: "conv2d/bias"
        string_val: "conv2d/kernel"
        string_val: "conv2d_1/bias"
        string_val: "conv2d_1/kernel"
        string_val: "conv2d_2/bias"
        string_val: "conv2d_2/kernel"
        string_val: "conv2d_3/bias"
        string_val: "conv2d_3/kernel"
        string_val: "conv2d_transpose/bias"
        string_val: "conv2d_transpose/kernel"
        string_val: "conv2d_transpose_1/bias"
        string_val: "conv2d_transpose_1/kernel"
        string_val: "conv2d_transpose_2/bias"
        string_val: "conv2d_transpose_2/kernel"
        string_val: "conv2d_transpose_3/bias"
        string_val: "conv2d_transpose_3/kernel"
        string_val: "training/Adam/Variable"
        string_val: "training/Adam/Variable_1"
        string_val: "training/Adam/Variable_10"
        string_val: "training/Adam/Variable_11"
        string_val: "training/Adam/Variable_12"
        string_val: "training/Adam/Variable_13"
        string_val: "training/Adam/Variable_14"
        string_val: "training/Adam/Variable_15"
        string_val: "training/Adam/Variable_16"
        string_val: "training/Adam/Variable_17"
        string_val: "training/Adam/Variable_18"
        string_val: "training/Adam/Variable_19"
        string_val: "training/Adam/Variable_2"
        string_val: "training/Adam/Variable_20"
        string_val: "training/Adam/Variable_21"
        string_val: "training/Adam/Variable_22"
        string_val: "training/Adam/Variable_23"
        string_val: "training/Adam/Variable_24"
        string_val: "training/Adam/Variable_25"
        string_val: "training/Adam/Variable_26"
        string_val: "training/Adam/Variable_27"
        string_val: "training/Adam/Variable_28"
        string_val: "training/Adam/Variable_29"
        string_val: "training/Adam/Variable_3"
        string_val: "training/Adam/Variable_30"
        string_val: "training/Adam/Variable_31"
        string_val: "training/Adam/Variable_32"
        string_val: "training/Adam/Variable_33"
        string_val: "training/Adam/Variable_34"
        string_val: "training/Adam/Variable_35"
        string_val: "training/Adam/Variable_36"
        string_val: "training/Adam/Variable_37"
        string_val: "training/Adam/Variable_38"
        string_val: "training/Adam/Variable_39"
        string_val: "training/Adam/Variable_4"
        string_val: "training/Adam/Variable_40"
        string_val: "training/Adam/Variable_41"
        string_val: "training/Adam/Variable_42"
        string_val: "training/Adam/Variable_43"
        string_val: "training/Adam/Variable_44"
        string_val: "training/Adam/Variable_45"
        string_val: "training/Adam/Variable_46"
        string_val: "training/Adam/Variable_47"
        string_val: "training/Adam/Variable_5"
        string_val: "training/Adam/Variable_6"
        string_val: "training/Adam/Variable_7"
        string_val: "training/Adam/Variable_8"
        string_val: "training/Adam/Variable_9"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  device: "/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 69
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  device: "/device:CPU:0"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_INT64
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Identity"
  op: "Identity"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp"
  op: "AssignVariableOp"
  input: "Adam/beta_1"
  input: "save/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_1"
  op: "Identity"
  input: "save/RestoreV2:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_1"
  op: "AssignVariableOp"
  input: "Adam/beta_2"
  input: "save/Identity_1"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_2"
  op: "Identity"
  input: "save/RestoreV2:2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_2"
  op: "AssignVariableOp"
  input: "Adam/decay"
  input: "save/Identity_2"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_3"
  op: "Identity"
  input: "save/RestoreV2:3"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "save/AssignVariableOp_3"
  op: "AssignVariableOp"
  input: "Adam/iterations"
  input: "save/Identity_3"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "save/Identity_4"
  op: "Identity"
  input: "save/RestoreV2:4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_4"
  op: "AssignVariableOp"
  input: "Adam/lr"
  input: "save/Identity_4"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_5"
  op: "Identity"
  input: "save/RestoreV2:5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_5"
  op: "AssignVariableOp"
  input: "conv2d/bias"
  input: "save/Identity_5"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_6"
  op: "Identity"
  input: "save/RestoreV2:6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_6"
  op: "AssignVariableOp"
  input: "conv2d/kernel"
  input: "save/Identity_6"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_7"
  op: "Identity"
  input: "save/RestoreV2:7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_7"
  op: "AssignVariableOp"
  input: "conv2d_1/bias"
  input: "save/Identity_7"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_8"
  op: "Identity"
  input: "save/RestoreV2:8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_8"
  op: "AssignVariableOp"
  input: "conv2d_1/kernel"
  input: "save/Identity_8"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_9"
  op: "Identity"
  input: "save/RestoreV2:9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_9"
  op: "AssignVariableOp"
  input: "conv2d_2/bias"
  input: "save/Identity_9"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_10"
  op: "Identity"
  input: "save/RestoreV2:10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_10"
  op: "AssignVariableOp"
  input: "conv2d_2/kernel"
  input: "save/Identity_10"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_11"
  op: "Identity"
  input: "save/RestoreV2:11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_11"
  op: "AssignVariableOp"
  input: "conv2d_3/bias"
  input: "save/Identity_11"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_12"
  op: "Identity"
  input: "save/RestoreV2:12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_12"
  op: "AssignVariableOp"
  input: "conv2d_3/kernel"
  input: "save/Identity_12"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_13"
  op: "Identity"
  input: "save/RestoreV2:13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_13"
  op: "AssignVariableOp"
  input: "conv2d_transpose/bias"
  input: "save/Identity_13"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_14"
  op: "Identity"
  input: "save/RestoreV2:14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_14"
  op: "AssignVariableOp"
  input: "conv2d_transpose/kernel"
  input: "save/Identity_14"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_15"
  op: "Identity"
  input: "save/RestoreV2:15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_15"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/bias"
  input: "save/Identity_15"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_16"
  op: "Identity"
  input: "save/RestoreV2:16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_16"
  op: "AssignVariableOp"
  input: "conv2d_transpose_1/kernel"
  input: "save/Identity_16"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_17"
  op: "Identity"
  input: "save/RestoreV2:17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_17"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/bias"
  input: "save/Identity_17"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_18"
  op: "Identity"
  input: "save/RestoreV2:18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_18"
  op: "AssignVariableOp"
  input: "conv2d_transpose_2/kernel"
  input: "save/Identity_18"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_19"
  op: "Identity"
  input: "save/RestoreV2:19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_19"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/bias"
  input: "save/Identity_19"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_20"
  op: "Identity"
  input: "save/RestoreV2:20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_20"
  op: "AssignVariableOp"
  input: "conv2d_transpose_3/kernel"
  input: "save/Identity_20"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_21"
  op: "Identity"
  input: "save/RestoreV2:21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_21"
  op: "AssignVariableOp"
  input: "training/Adam/Variable"
  input: "save/Identity_21"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_22"
  op: "Identity"
  input: "save/RestoreV2:22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_22"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_1"
  input: "save/Identity_22"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_23"
  op: "Identity"
  input: "save/RestoreV2:23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_23"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_10"
  input: "save/Identity_23"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_24"
  op: "Identity"
  input: "save/RestoreV2:24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_24"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_11"
  input: "save/Identity_24"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_25"
  op: "Identity"
  input: "save/RestoreV2:25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_25"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_12"
  input: "save/Identity_25"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_26"
  op: "Identity"
  input: "save/RestoreV2:26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_26"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_13"
  input: "save/Identity_26"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_27"
  op: "Identity"
  input: "save/RestoreV2:27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_27"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_14"
  input: "save/Identity_27"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_28"
  op: "Identity"
  input: "save/RestoreV2:28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_28"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_15"
  input: "save/Identity_28"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_29"
  op: "Identity"
  input: "save/RestoreV2:29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_29"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_16"
  input: "save/Identity_29"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_30"
  op: "Identity"
  input: "save/RestoreV2:30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_30"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_17"
  input: "save/Identity_30"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_31"
  op: "Identity"
  input: "save/RestoreV2:31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_31"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_18"
  input: "save/Identity_31"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_32"
  op: "Identity"
  input: "save/RestoreV2:32"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_32"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_19"
  input: "save/Identity_32"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_33"
  op: "Identity"
  input: "save/RestoreV2:33"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_33"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_2"
  input: "save/Identity_33"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_34"
  op: "Identity"
  input: "save/RestoreV2:34"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_34"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_20"
  input: "save/Identity_34"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_35"
  op: "Identity"
  input: "save/RestoreV2:35"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_35"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_21"
  input: "save/Identity_35"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_36"
  op: "Identity"
  input: "save/RestoreV2:36"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_36"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_22"
  input: "save/Identity_36"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_37"
  op: "Identity"
  input: "save/RestoreV2:37"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_37"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_23"
  input: "save/Identity_37"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_38"
  op: "Identity"
  input: "save/RestoreV2:38"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_38"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_24"
  input: "save/Identity_38"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_39"
  op: "Identity"
  input: "save/RestoreV2:39"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_39"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_25"
  input: "save/Identity_39"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_40"
  op: "Identity"
  input: "save/RestoreV2:40"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_40"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_26"
  input: "save/Identity_40"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_41"
  op: "Identity"
  input: "save/RestoreV2:41"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_41"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_27"
  input: "save/Identity_41"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_42"
  op: "Identity"
  input: "save/RestoreV2:42"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_42"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_28"
  input: "save/Identity_42"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_43"
  op: "Identity"
  input: "save/RestoreV2:43"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_43"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_29"
  input: "save/Identity_43"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_44"
  op: "Identity"
  input: "save/RestoreV2:44"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_44"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_3"
  input: "save/Identity_44"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_45"
  op: "Identity"
  input: "save/RestoreV2:45"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_45"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_30"
  input: "save/Identity_45"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_46"
  op: "Identity"
  input: "save/RestoreV2:46"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_46"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_31"
  input: "save/Identity_46"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_47"
  op: "Identity"
  input: "save/RestoreV2:47"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_47"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_32"
  input: "save/Identity_47"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_48"
  op: "Identity"
  input: "save/RestoreV2:48"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_48"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_33"
  input: "save/Identity_48"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_49"
  op: "Identity"
  input: "save/RestoreV2:49"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_49"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_34"
  input: "save/Identity_49"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_50"
  op: "Identity"
  input: "save/RestoreV2:50"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_50"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_35"
  input: "save/Identity_50"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_51"
  op: "Identity"
  input: "save/RestoreV2:51"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_51"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_36"
  input: "save/Identity_51"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_52"
  op: "Identity"
  input: "save/RestoreV2:52"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_52"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_37"
  input: "save/Identity_52"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_53"
  op: "Identity"
  input: "save/RestoreV2:53"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_53"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_38"
  input: "save/Identity_53"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_54"
  op: "Identity"
  input: "save/RestoreV2:54"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_54"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_39"
  input: "save/Identity_54"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_55"
  op: "Identity"
  input: "save/RestoreV2:55"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_55"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_4"
  input: "save/Identity_55"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_56"
  op: "Identity"
  input: "save/RestoreV2:56"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_56"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_40"
  input: "save/Identity_56"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_57"
  op: "Identity"
  input: "save/RestoreV2:57"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_57"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_41"
  input: "save/Identity_57"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_58"
  op: "Identity"
  input: "save/RestoreV2:58"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_58"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_42"
  input: "save/Identity_58"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_59"
  op: "Identity"
  input: "save/RestoreV2:59"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_59"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_43"
  input: "save/Identity_59"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_60"
  op: "Identity"
  input: "save/RestoreV2:60"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_60"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_44"
  input: "save/Identity_60"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_61"
  op: "Identity"
  input: "save/RestoreV2:61"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_61"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_45"
  input: "save/Identity_61"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_62"
  op: "Identity"
  input: "save/RestoreV2:62"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_62"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_46"
  input: "save/Identity_62"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_63"
  op: "Identity"
  input: "save/RestoreV2:63"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_63"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_47"
  input: "save/Identity_63"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_64"
  op: "Identity"
  input: "save/RestoreV2:64"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_64"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_5"
  input: "save/Identity_64"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_65"
  op: "Identity"
  input: "save/RestoreV2:65"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_65"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_6"
  input: "save/Identity_65"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_66"
  op: "Identity"
  input: "save/RestoreV2:66"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_66"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_7"
  input: "save/Identity_66"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_67"
  op: "Identity"
  input: "save/RestoreV2:67"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_67"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_8"
  input: "save/Identity_67"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/Identity_68"
  op: "Identity"
  input: "save/RestoreV2:68"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/AssignVariableOp_68"
  op: "AssignVariableOp"
  input: "training/Adam/Variable_9"
  input: "save/Identity_68"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/AssignVariableOp"
  input: "^save/AssignVariableOp_1"
  input: "^save/AssignVariableOp_10"
  input: "^save/AssignVariableOp_11"
  input: "^save/AssignVariableOp_12"
  input: "^save/AssignVariableOp_13"
  input: "^save/AssignVariableOp_14"
  input: "^save/AssignVariableOp_15"
  input: "^save/AssignVariableOp_16"
  input: "^save/AssignVariableOp_17"
  input: "^save/AssignVariableOp_18"
  input: "^save/AssignVariableOp_19"
  input: "^save/AssignVariableOp_2"
  input: "^save/AssignVariableOp_20"
  input: "^save/AssignVariableOp_21"
  input: "^save/AssignVariableOp_22"
  input: "^save/AssignVariableOp_23"
  input: "^save/AssignVariableOp_24"
  input: "^save/AssignVariableOp_25"
  input: "^save/AssignVariableOp_26"
  input: "^save/AssignVariableOp_27"
  input: "^save/AssignVariableOp_28"
  input: "^save/AssignVariableOp_29"
  input: "^save/AssignVariableOp_3"
  input: "^save/AssignVariableOp_30"
  input: "^save/AssignVariableOp_31"
  input: "^save/AssignVariableOp_32"
  input: "^save/AssignVariableOp_33"
  input: "^save/AssignVariableOp_34"
  input: "^save/AssignVariableOp_35"
  input: "^save/AssignVariableOp_36"
  input: "^save/AssignVariableOp_37"
  input: "^save/AssignVariableOp_38"
  input: "^save/AssignVariableOp_39"
  input: "^save/AssignVariableOp_4"
  input: "^save/AssignVariableOp_40"
  input: "^save/AssignVariableOp_41"
  input: "^save/AssignVariableOp_42"
  input: "^save/AssignVariableOp_43"
  input: "^save/AssignVariableOp_44"
  input: "^save/AssignVariableOp_45"
  input: "^save/AssignVariableOp_46"
  input: "^save/AssignVariableOp_47"
  input: "^save/AssignVariableOp_48"
  input: "^save/AssignVariableOp_49"
  input: "^save/AssignVariableOp_5"
  input: "^save/AssignVariableOp_50"
  input: "^save/AssignVariableOp_51"
  input: "^save/AssignVariableOp_52"
  input: "^save/AssignVariableOp_53"
  input: "^save/AssignVariableOp_54"
  input: "^save/AssignVariableOp_55"
  input: "^save/AssignVariableOp_56"
  input: "^save/AssignVariableOp_57"
  input: "^save/AssignVariableOp_58"
  input: "^save/AssignVariableOp_59"
  input: "^save/AssignVariableOp_6"
  input: "^save/AssignVariableOp_60"
  input: "^save/AssignVariableOp_61"
  input: "^save/AssignVariableOp_62"
  input: "^save/AssignVariableOp_63"
  input: "^save/AssignVariableOp_64"
  input: "^save/AssignVariableOp_65"
  input: "^save/AssignVariableOp_66"
  input: "^save/AssignVariableOp_67"
  input: "^save/AssignVariableOp_68"
  input: "^save/AssignVariableOp_7"
  input: "^save/AssignVariableOp_8"
  input: "^save/AssignVariableOp_9"
}
versions {
  producer: 26
}
