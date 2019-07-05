---
layout: keynote
title: "Tensorflow Estimator 简介"
subtitle: 'Using a high level tool of tensorflow'
author: "OneGame"
header-style: text
categories:
  - tools
tags:
  - tensorflow
  - Estimator
---

>tf.estimator属于tensorflow中的高级抽象封装，目的是为了提供开发着的开发速度，但是同时也会在一定程度上限制灵活性。  


---

# 简介
tf.estimator属于tensorflow中的高级抽象封装，目的是为了提供开发着的开发速度，但是同时也会在一定程度上限制灵活性。  
![tensorflow编程栈](https://upload-images.jianshu.io/upload_images/9550643-43922525245f9a27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



Estimator封装了训练、评估、预测及导出等操作
## 如何定义一个Estimator
Tensorflow为我们预定义好了一些Estimator（如下图），我们也可根据自己的网络结构去自定义Estimator。
![estimator_types.png](https://upload-images.jianshu.io/upload_images/9550643-3934e79e4eec2bb1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通常来说我们去实例化一个Estimator并且进行训练只需要两步：
```python
estimator = tf.estimator.Estimator(..) #实例化
esitmator.train(..) #训练
estimator.evaluate(..) #评估
estimator.predict(..) #预测
estimator.save(..) #保存、导出
estimator.load(..) #加载
```
所以接下来我们分别看一下每一步中都需要我们准备什么。  
*** 
**实例化Estimator**
```python
#tf.estimator.Estimator的构造函数
__init__(
    model_fn,
    model_dir=None,
    config=None,
    params=None,
    warm_start_from=None
)
```
**其中：**
 * model_fn ：模型定义函数指针，用来定义网络结构，其定义如下：
```
def my_model_fn(
   features, # This is batch_features from input_fn, passed by estimator
   labels,   # This is batch_labels from input_fn, passed by estimator
   mode,     # An instance of tf.estimator.ModeKeys,e.g. train、eval or predict
   params):  # Additional configuration
```
 **features**，**labels** 是在模型训练（或者评估和预测）过程中通过`input_fn`的返回传入的  
**mode**是指定模型的运行模型，如训练、评估和预测  
**params**是其他参数用于模型的构建和超参的调优
**注：** 该函数必须返回一个[`tf.estimator.EstimatorSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)的实例
  ```python
  @staticmethod
__new__(
    cls,
    mode,
    predictions=None,
    loss=None,
    train_op=None,
    eval_metric_ops=None,
    export_outputs=None,
    training_chief_hooks=None,
    training_hooks=None,
    scaffold=None,
    evaluation_hooks=None,
    prediction_hooks=None
)
  ```
* model_dir: 是模型及checkpoint相关文件的存储目录
* config: estimator.RunConfig configuration object
* params：model_fn的params
* warm_start_from：Optional string filepath to a checkpoint or SavedModel to warm-start from

***
**训练模型**
接下来我们看一下estimitor的训练都需要什么，函数的原型定义如果下：
```python
train(
    input_fn,
    hooks=None,
    steps=None,
    max_steps=None,
    saving_listeners=None
)
```
* input_fn: 数据封装及转换函数，用来将原始数据封装成tf.data.Dataset输出提供给train（或者eval和predict使用），并且提供数据迭代和permutation等功能。其定义如下：
```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
```
* hooks: List of [`tf.train.SessionRunHook`](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) subclass instances,用来在训练等过程中进行回调。`todo:补充例子`
* steps: 迭代次数，如果是None训练次数有input_fn生成数据控制
* max_steps：最大迭代次数，区别于steps是，两次调用train(steps=100)会训练200步，两次调用train(max_steps=100)只会训练100步

***
**评估模型**
***
**预测**
***
**保存、导出模型**
***
**模型的加载**
***

综上，当我们使用Estimator进行建模训练的步骤如下：
##### 1、定义网络结构
如果使用tensorflow为我们预定义好的网络则可略过此步，否则需要定义在estimator构造阶段提到的model_fn方法，在该方法中我们要实现的是：
* **定义模型**
基本的深度学习模型必须要定义*输入层*、*隐层*和*输出层*：
  * 输入层
  ```
    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
  ```
    上面的行会应用特征列定义的转换，从而创建模型的输入层。
   * 隐藏层
   ```
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
   ```
   * 输出层
   ```
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
   ```
* **为训练、预测和评估分别指定相应的计算**  

| Estimator 方法 | Estimator 模式(mode) | tf.estimator.EstimatorSpec必须参数|
| :-: | :-: |:-:|
| train()| ModeKeys.TRAIN |loss和train_op|
|predict()|ModeKeys.PREDICT|loss|
|eval()|ModeKeys.EVAL| predictions|

如下是一个简单的例子：
```
def my_model_fn(features, labels, mode):
  if (mode == tf.estimator.ModeKeys.TRAIN or
      mode == tf.estimator.ModeKeys.EVAL):
    loss = ...
  else:
    loss = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = ...
  else:
    train_op = None
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = ...
  else:
    predictions = None

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op)
```




##### 2、 编写一个或多个数据集导入函数

```
def input_fn(dataset):
   ...  # manipulate dataset, extracting the feature dict and the label
   return feature_dict, label
```

##### 3、定义特征列
对特征的定义主要由 [`tf.feature_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column)实现，它标识了特征名称、特征类型和任何输入预处理操作。
```
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
```


![inputs_to_model_bridge.jpg](https://upload-images.jianshu.io/upload_images/9550643-47e415bc23ec99c7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**tf.feature_column**在dataset和estimator之间起到了桥接作用，将dataset中的原始特征转换成模型可以使用的特征形式。  

根据不同的特征类型，tf.feature_column又包含如下9个不同的函数用来处理不同的数据类型：
![feature_columns类型](https://upload-images.jianshu.io/upload_images/9550643-be0278f30cdc42be.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
总体上特征可以分为Dense特征（Dense Column）和Sparse特征（Categorical Column），最终所有的特征列转换为Dense Column作为网络的输入。

**对于Dense Column：**  
* **numeric_column**  
数值列，用来处理连续值
```
tf.feature_column.numeric_column(
    key,
    shape=(1,),
    default_value=None,
    dtype=tf.float32,
    normalizer_fn=None
)
```

* **indicator_column**  
指标列,指标列将每个类别视为独热矢量中的一个元素，其中匹配类别的值为 1，其余类别为 0,即将Categorical_column的输出做度热编码
```
tf.feature_column.indicator_column(categorical_column)
```

* **embedding_column**  
类似上个，只是目标空间为分布式连续空间
```
categorical_column = ... # Create any categorical column

# Represent the categorical column as an embedding column.
# This means creating an embedding vector lookup table with one element for each category.
# dimension选择经验公式
# embedding_dimensions =  number_of_categories**0.25
embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=embedding_dimensions)
```

**对于Categorical Column：**    
* **categorical_column_with_identity**   
```
tf.feature_column.categorical_column_with_identity(
    key,
    num_buckets,
    default_value=None
)
```
所有的输入都必须在[0, num_buckets)内，如果超出范围，则会使用default_value，如果default_value没有定义则会抛异常。

* **categorical_column_with_vocabulary_list**  
将string or integer转换为id
```
tf.feature_column.categorical_column_with_vocabulary_list(
    key,
    vocabulary_list,
    dtype=None,
    default_value=-1,
    num_oov_buckets=0
)
```

* **categorical_column_with_vocabulary_list**  
功能同上，只是使用文件做词表

* **categorical_column_with_hash_bucket**  
进行hash分类，限制类别数量
```
tf.feature_column.categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=tf.string
)
```
![hashed_column.jpg](https://upload-images.jianshu.io/upload_images/9550643-9469fa479491124f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* **crossed_column**  
特征组合列，通过将多个特征组合为一个特征
```
tf.feature_column.crossed_column(
    keys,
    hash_bucket_size,
    hash_key=None
)
#e.g
# Bucketize the latitude and longitude using the `edges`
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges))

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(atlanta.longitude.edges))

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)
```
**其他**
* ** bucketized_column**
分桶列，顾名思义是用来分桶的
```
 tf.feature_column.bucketized_column(
    source_column,
    boundaries
)
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```

##### 4、实例化相关的预创建的 Estimator
```
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

##### 5、调用训练、评估或推理方法
```
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```
