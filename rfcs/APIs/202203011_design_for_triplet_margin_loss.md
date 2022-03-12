# paddle.nn.TripletMarginLoss 设计文档

| API 名称     | paddle.nn.TripletMarginLoss                   |
| ------------ | --------------------------------------------- |
| 提交作者     | Ainavo                                        |
| 提交时间     | 2022-03-11                                   |
| 版本号       | V1.0                                          |
| 依赖飞桨版本 | v2.2.0                                        |
| 文件名       | 202203011_design_for_triplet_margin_loss.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算领域 API，Paddle 需要扩充 API`paddle.nn.TripletMarginLoss`。

## 2、功能目标

增加 API`paddle.nn.TripletMarginLoss`，实现三元损失函数。

## 3、意义

丰富 paddle 中的 loss 库，增加三元损失函数 API。

# 二、飞桨现状

目前 paddle 缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch 中有 API`torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`，以及对应的`torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`.在 pytorch 中，介绍为：

Creates a criterion that measures the triplet loss given an input tensors $x 1, x 2, x 3$ and a margin with a value greater than 0 . This is used for measuring a relative similarity between samples. A triplet is composed by a, $p$ and $n$ (i.e., anchor, positive examples and negative examples respectively). The shapes of all input tensors should be $(N, D)$.

The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
The loss function for each sample in the mini-batch is:

$$
L(a, p, n)=\max \left\{d\left(a_{i}, p_{i}\right)-d\left(a_{i}, n_{i}\right)+\operatorname{margin}, 0\right\}
$$

where

$$
d\left(x_{i}, y_{i}\right)=\left\|\mathbf{x}_{i}-\mathbf{y}_{i}\right\|_{p}
$$

### 实现方法

在实现方法上, Pytorch 是通过 c++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/701fa16eed40c633d8eef6b4f04ab73a75c24749/aten/src/ATen/native/Loss.cpp?q=triplet_margin_loss#L148)。
c++代码实现如下：

```c++
Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  auto a_dim = anchor.dim();
  auto p_dim = positive.dim();
  auto n_dim = negative.dim();
  TORCH_CHECK(
      a_dim == p_dim && p_dim == n_dim,
      "All inputs should have same dimension but got ",
      a_dim,
      "D, ",
      p_dim,
      "D and ",
      n_dim,
      "D inputs.")
  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}
```

整体逻辑为：

- 检查输入` anchor``positive``negative `的维度是否相等，不等报错
- 通过`pairwise_distance()`函数，分别计算` anchor``positive `之间的距离，` anchor``negative `之间的距离。
- `swap`参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
- 通过`clamp_distance()`实现核心公式，计算出`loss`.
- `apply_loss_redution()`函数选择输出的方式包括（` mean``sum `等）

## TensorFlow

### 实现方法

在实现方法上 tensorflow 以 python API 组合实现，[代码位置](https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/delf/delf/python/training/losses/ranking_losses.py).
其中核心代码为：

```Python
def triplet_loss(queries, positives, negatives, margin=0.1):
  """Calculates Triplet Loss.
  Triplet loss tries to keep all queries closer to positives than to any
  negatives. Differently from the Contrastive Loss, Triplet Loss uses squared
  distances when computing the loss.
  Args:
    queries: [batch_size, dim] Anchor input tensor.
    positives: [batch_size, dim] Positive sample input tensor.
    negatives: [batch_size, num_neg, dim] Negative sample input tensor.
    margin: Float triplet loss loss margin.
  Returns:
    loss: Scalar tensor.
  """
  dim = tf.shape(queries)[1]
  # Number of `queries`.
  batch_size = tf.shape(queries)[0]
  # Number of `negatives`.
  num_neg = tf.shape(negatives)[1]

  # Preparing negatives.
  stacked_negatives = tf.reshape(negatives, [num_neg * batch_size, dim])

  # Preparing queries for further loss calculation.
  stacked_queries = tf.repeat(queries, num_neg, axis=0)

  # Preparing positives for further loss calculation.
  stacked_positives = tf.repeat(positives, num_neg, axis=0)

  # Computes *squared* distances.
  distance_positives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_positives), axis=1)
  distance_negatives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_negatives), axis=1)
  # Final triplet loss calculation.
  loss = tf.reduce_sum(
      tf.maximum(distance_positives - distance_negatives + margin, 0.0))
  return loss
```

整体逻辑为：

- 读取输入`queries`的维度、`batch_size`大小。
- 读取输入`negatives`的数量并`reshape()`成`[num_neg * batch_size, dim]`的形状，对`positives`进行相同操作。
- 通过`tf.square()`函数计算欧式距离，`tf.reduce_sum()`函数沿第二根轴求和，分别得到`distance_positives`和`distance_negatives`
- 通过`tf.maximum()`实现核心公式，计算出 loss

# 四、对比分析

- 使用场景与功能：Pytorch实现求解三元组API的基本功能，TensorFlow以训练中的实际参数为代入，两种代码风格不同。功能上基本一致，这里paddle三元组API的设计将对齐Pytorch中的三元组API。

# 五、方案设计

## 命名与参数设计

API 设计为`paddle.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`及`padde.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`
命名与参数顺序为：形参名`anchor`->`tensor`和`positive`->`tensor`,`negative`->`tensor` ,其余与 paddle 其他 API 保持一致性，不影响实际功能使用。

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

主要按下列步骤进行组合实现,实现位置`Paddle\python\paddle\nn\functional\distance.py`与`Paddle\python\paddle\nn\layer\loss.py`等loss方法放在一起：

1. 使用`check_variable_and_dtype`检查输入的维度是否对齐。
2. `in_dynamic_mode`检查各个输入参数是否符合规范。
3. 使用`pairwise_distance()`分别计算得到正锚点与样本和负锚点与样本的距离。
4. `swap`参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
5. 通过`paddle.clip()`实现公式所示求出得loss。
6. 根据`reduction`的输入参数，选择loss的输出方式。


# 六、测试和验收的考量

测试考虑的 case 如下：

- `padde.nn.functional.triplet_margin_loss`,``paddle.nn.TripletMarginLoss`和torch结果是否一致；
- 参数`margin`为 float 和 1-D Tensor 时输出的正确性；
- 参数`p`为的正确性
- `eps`参数的正确性；
- `swap`参数的正确性；
- 输入含`NaN`结果的正确性；
- `reduction`对应不同参数的正确性；
- 输入维度不同，报错的正确性；


# 七、可行性分析及规划排期

方案主要依赖现有 paddle api 组合而成，且依赖的`paddle.nn.layer.pairwise_distance`、`paddle.clip`、`paddle.min`已于前期合入，由于`paddle.nn.layer.pairwise_distance`代码将求距离的实现方法写在`nn.layer`中，且与常规`nn.layer`中代码风格不同，故依据其余代码风格将`paddle.nn.layer.pairwise_distance`重构到`paddle.nn.functional.pairwise_distance`中，并通过`paddle.nn.layer.loss`进行调用，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无 

# 附件及参考资料

无
