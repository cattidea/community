# paddle.nn.functional.pairwise_distance 设计文档

| API 名称     | paddle.nn.functional.pairwise_distance   |
| ------------ | ---------------------------------------- |
| 提交作者     | Ainavo                                   |
| 提交时间     | 2022-07-04                               |
| 版本号       | V1.0                                     |
| 依赖飞桨版本 | v2.2.3                                   |
| 文件名       | 20220704_design_for_pairwise_distance.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持神经网络搭建相关 API，Paddle 需要扩充 API `paddle.nn.functional.pairwise_distance` 。

## 2、功能目标

增加 API `paddle.nn.functional.pairwise_distance` ，用于计算两组向量两两之间的距离。

## 3、意义

增加 paddle 中的距离计算。

# 二、飞桨现状

目前 paddle 包含 `paddle.linalg.norm` 和 `paddle.dist`，并有。

# 三、业内方案调研

## Pytorch

Pytorch 中有 functional API `torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False) → Tensor`，以及对应的 Module `torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)`

在 Pytorch 中，介绍为：

> Computes the pairwise distance between vectors :math: $v_1$, :math: $v_2$ using the p-norm:

>$$ 
\Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}. 
>$$

### 实现方法

在实现方法上，Pytorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/9e137ee583c4fdb2dd3aa0c425dc9c289454cbf2/aten/src/ATen/native/Distance.cpp)。
C++ 代码实现如下：

```c++
Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  // Since either x1 or x2 could be broadcasted
  auto x1_dim = x1.dim();
  auto x2_dim = x2.dim();
  auto output_dim = x1_dim > x2_dim ? x1_dim : x2_dim;
  auto innermost_dim = output_dim - 1;
  return at::norm(x1 - x2 + eps, p, innermost_dim, keepdim);
}

TORCH_IMPL_FUNC(norm_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, c10::nullopt, result);
}

TORCH_IMPL_FUNC(norm_dtype_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, dtype, result);
}

void impl_func_norm(
    const Tensor& self,
    const OptionalScalarRef& opt_p,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    const Tensor& result) {
  auto p = opt_p.has_value() ? opt_p.get() : Scalar(2.0).to<double>();
  auto in_dtype = opt_dtype.value_or(self.scalar_type());
  auto out_dtype = result.scalar_type();

  // See the note [Reductions do not use vectorized ops]
  Tensor self_;
  if (self.is_cpu() && self.is_complex() && std::abs(p.toDouble()) == INFINITY) {
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype).abs();
    } else {
      self_ = self.abs();
    }
  } else {
    self_ = self;
  }


  // omit in_dtype in the following call, to avoid make_reduction explicitly
  // casting input to out_dtype
  auto iter = isComplexType(self_.scalar_type())
      ? meta::make_reduction(self_, result, dim, keepdim, in_dtype)
      : meta::make_reduction_from_out_ty(self_, result, dim, keepdim, out_dtype);

  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter.device_type(), iter, p);
  }
}

```

整体逻辑为：

- 获取 x1、x2 的维度
- 输出维度默认为输入中维度大的向量
- innermost_dim：输入维度 -1
- 调用 norm 算子，norm 算子的逻辑：
## TensorFlow
TensorFlow 中有Model`nsl.keras.layers.PairwiseDistance(distance_config=None, **kwargs)`以及距离函数`nsl.lib.pairwise_distance_wrapper(sources, targets, weights=1.0, distance_config=None)`

### 实现方法

在实现方法上 tensorflow 以 python API 组合实现，[代码位置](https://github.com/tensorflow/neural-structured-learning/blob/c21dad4feff187cdec041a564193ea7b619b8906/neural_structured_learning/lib/distances.py#L222)。

其中核心代码为：

Python 代码实现如下：
```python
def pairwise_distance_wrapper(sources,
                              targets,
                              weights=1.0,
                              distance_config=None):
  """A wrapper to compute the pairwise distance between `sources` and `targets`.
  `distances = weights * distance_config.distance_type(sources, targets)`
  This wrapper calculates the weighted distance between `(sources, targets)`
  pairs, and provides an option to return the distance as the sum over the
  difference along the given axis, when vector based distance is needed.
  For the usage of `weights` and `reduction`, please refer to `tf.losses`. For
  the usage of `sum_over_axis`, see the following examples:
  Given target tensors with shape `[batch_size, features]`, the reduction set to
  `tf.compat.v1.losses.Reduction.MEAN`, and `sum_over_axis` set to the last
  dimension, the weighted average distance of sample pairs will be returned.
  For example: With a distance_config('L2', sum_over_axis=-1), the distance
  between [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]]
  will be {(0+0) + (4+0) + (16+4) + (16+1)}/4 = 10.25
  If `sum_over_axis` is `None`, the weighted average distance of feature pairs
  (instead of sample pairs) will be returned. For example: With a
  distance_config('L2'), the distance between
  [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]] will be
  {(0+0) + (4+0) + (16+4) + (16+1)}/8 = 5.125
  If `transform_fn` is not `None`, the transform function is applied to both
  `sources` and `targets` before computing the distance. For example:
  `distance_config('KL_DIVERGENCE', sum_over_axis=-1, transform_fn='SOFTMAX')`
  treats `sources` and `targets` as logits, and computes the KL-divergence
  between the two probability distributions.
  Args:
    sources: `Tensor` of type `float32` or `float64`.
    targets: `Tensor` of the same type and shape as `sources`.
    weights: (optional) `Tensor` whose rank is either 0, or the same as that of
      `targets`, and must be broadcastable to `targets` (i.e., all dimensions
      must be either `1`, or the same as the corresponding distance dimension).
    distance_config: An instance of `nsl.configs.DistanceConfig` that contains
      the following configuration (or hyperparameters) for computing distances:
      (a) `distance_type`: Type of distance function to apply.
      (b) `reduction`: Type of distance reduction. See `tf.losses.Reduction`.
      (c) `sum_over_axis`: (optional) The distance is the sum over the
        difference along the specified axis. Note that if `sum_over_axis` is not
        `None` and the rank of `weights` is non-zero, then the size of `weights`
        along `sum_over_axis` must be 1.
      (d) `transform_fn`: (optional) If set, both `sources` and `targets` will
        be transformed before calculating the distance. If set to 'SOFTMAX', it
        will be performed on the axis specified by 'sum_over_axis', or -1 if the
        axis is not specified. If `None`, the default distance config will be
        used.
  Returns:
    Weighted distance scalar `Tensor`. If `reduction` is
      `tf.compat.v1.losses.Reduction.MEAN`, this has the same shape as
      `targets`.
  Raises:
    ValueError: If the shape of targets doesn't match that of sources, or if the
      shape of weights is invalid.
    TypeError: If the distance function gets an unexpected keyword argument.
  """
  if distance_config is None:
    distance_config = configs.DistanceConfig()  # Default configs.

  tf.compat.v1.losses.Reduction.validate(distance_config.reduction)

  if distance_config.transform_fn is not configs.TransformType.NONE:
    sources = _apply_transform(sources, distance_config.transform_fn,
                               distance_config.sum_over_axis)
    targets = _apply_transform(targets, distance_config.transform_fn,
                               distance_config.sum_over_axis)

  sum_over_axis = distance_config.sum_over_axis
  # Validates the `sum_over_axis`
  _assert_valid_axis(sources.get_shape().ndims, sum_over_axis)
  distance_fn = _select_distance_fn(distance_config.distance_type)
  if distance_config.distance_type == configs.DistanceType.COSINE:
    # Cosine distance function assumes input tensors have been unit-normalized
    sources = tf.nn.l2_normalize(sources, axis=sum_over_axis)
    targets = tf.nn.l2_normalize(targets, axis=sum_over_axis)
  if _is_axis_required_in_distance_fn(distance_config.distance_type):
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        axis=sum_over_axis,
        reduction=distance_config.reduction,
        loss_collection=None)
  else:
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        reduction=distance_config.reduction,
        loss_collection=None)
    if sum_over_axis is not None and _is_reduced_by_average(
        distance_config.reduction):
      # The distance is divided by the size of targets tensor, so we need to
      # rescale the distance by multiplying the size of axis. Note, the distance
      # function with `axis` as a required argument (e.g., consine distance)
      # does not need to be rescaled.
      weights = tf.convert_to_tensor(value=weights)
      weights_shape = weights.get_shape().as_list()
      if weights_shape and weights_shape[sum_over_axis] != 1:
        raise ValueError('Shape of weights along the axis %d must be 1.' %
                         sum_over_axis)
      distances *= sources.shape.dims[sum_over_axis].value
  return distances
```
参数表：
- `sources` : Tensor ( float32 或 float64 )  
- `targets` : Tensor (与 `sources` 类型保持一致)  
- `weights` : (optional) 维度为 1 或 与得到的距离维度一致，必须保证可以广播到 `targets`
- `distance_config` : 计算距离的配置（或超参数） `nsl.configs.DistanceConfig` 实例包含以下配置：(a) `distance_type` :要应用的距离函数的类型。(b) `reduction` :距离减少的类型。(c) `sum_over_axis` :（可选）距离是沿指定轴的差值之和。注：该参数如果不是 `None` 并且 `weights` 的秩不为零，则 `weights` 沿着 `sum_over_axis` 的大小必须为 1 。 (d) `transform_fn` :（可选）如果设置，则 `sources` 和 `targets` 都将在计算距离之前进行转换。如果设置为`SOFTMAX` ，它将在 `sum_over_axis` 指定的轴上执行，如果未指定轴，则为 -1 。如果是 `None` ，将使用默认距离配置。
  
整体逻辑为：
- 加载距离配置 `distance_config` ， 如果是 `None` ， 则使用默认配置。
- 加载距离配置中参数 `distance_config.transform_fn` ,如果不是 `None` ，则分别对 `sources` 和 `targets` 应用 `_apply_transform` 。
- 加载距离配置中参数 `distance_config.distance_type` ,如果不是 `None` ，则分别对 `sources` 和 `targets` 应用 `tf.nn.l2_normalize` 。
- 判断求解的距离类型，对应使用 `distance_fn` 函数（这里不展开说明）。
- 将求解得到的距离进行加权操作。

# 四、对比分析

- 使用场景与功能：Pytorch 实现计算两组向量两两之间的距离 API 的基本功能，TensorFlow 以训练中的实际参数为代入，两种代码风格不同。功能上基本一致，这里 paddle 计算两组向量两两之间的距离 API 的设计将对齐 Pytorch 中相应的 API。

# 五、方案设计

## 命名与参数设计

共添加以下 API ：

- `padde.nn.functional.pairwise_distance(x, y, p=2.0, epsilon=1e-06, keepdim=False, name=None) -> Tensor` （后续详述为何要添加此 API）

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

### nn.functional.pairwise_distance

该 API 实现于 `Paddle\python\paddle\nn\functional\distance.py`（目前尚无该文件，故需要新建）

paddle 目前没有 `pairwise_distance` 这样的 functional API，只有 `nn.PairwiseDistance` 这一 Layer API，不方便复用，因此先将 `nn.PairwiseDistance` API 的计算逻辑提取到 `nn.functional.pairwise_distance` 并暴露（已经调研过 torch 也有 `torch.nn.functional.pairwise_distance` 这样的 functional API）

实现逻辑同现有的 `nn.PairwiseDistance`，只不过按照 functional 的风格来写。




# 六、测试和验收的考量

测试考虑的 case 如下：

- `paddle.nn.functional.pairwise_distance` 和 torch 结果是否一致；
- 参数 `margin` 为 float 和 1-D Tensor 时输出的正确性；
- 参数 `p` 各个取值的正确性；
- 参数 `epsilon` 的正确性；
- 参数 `swap` 为 `True` 或者 `False` 的正确性；
- 输入含 `NaN` 结果的正确性；
- `reduction` 对应不同参数的正确性；
- 错误检查：`p` 值 `p<1` 时能正确抛出错误

# 七、可行性分析及规划排期

方案主要依赖现有 paddle api 组合而成，且依赖的 `paddle.clip`、`paddle.min` 已于前期合入，依赖的 `paddle.nn.functional.pairwise_distance` 从 `paddle.nn.PairwiseDistance` 提取得到。

具体规划为

- 阶段一：提取 `nn.PairwiseDistance` 主要逻辑到 `nn.functional.pairwise_distance`，在 `nn.PairwiseDistance` 中调用它，保证其逻辑不变
- 阶段二：完成 `nn.functioanl.triplet_margin_loss`，并在 `nn.TripletMarginLoss` 中调用
- 阶段三：完成 `nn.functioanl.triplet_margin_loss` 单元测试
- 阶段四：为三个新的 API 书写中文文档

# 八、影响面

除去本次要新增的两个 API，额外增加了一个 `nn.functional.pairwise_distance`，但对原有的 `nn.PairwiseDistance` 没有影响

# 名词解释

无

# 附件及参考资料

无