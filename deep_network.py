import tensorflow as tf
import tensorflow.contrib.slim as slim


num_columns=24
num_classes=4


def model(mode=tf.estimator.ModeKeys.TRAIN):


    ground_truth = tf.placeholder(tf.float32,[None,4], name = 'GroundTruthInputPlaceholder') #onehotencoded with depth 4

    input_tensor = tf.placeholder(tf.float32,[None,num_columns],name = 'BottlenecksPlaceholder')  #num_columns=18 keypoint features
    final_tensor_name = 'SoftmaxLayer'

    with tf.name_scope('Network'):
      WI = tf.truncated_normal_initializer(mean=0,stddev=0.001)
      BI = tf.zeros_initializer()
      with slim.arg_scope([slim.fully_connected],weights_initializer = WI, biases_initializer = BI ,trainable = True):
        input_tensor = slim.flatten(input_tensor)
        print(input_tensor.shape)
        net = slim.fully_connected(input_tensor,128,scope='fc1')
        print(net.shape)
        net = slim.fully_connected(net,128,scope='fc2')
        print(net.shape)
        logits = slim.fully_connected(net,num_classes,activation_fn=None,scope='fc3')
        print(logits.shape)
        #net = slim.flatten(net,scope = 'flat')
    #logits = tf.nn.softmax(net, name=final_tensor_name)
    #print(logits.shape)


    predicted_classes = tf.argmax(logits, 1)

    # Create prediction ops.
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
      'class': predicted_classes,
      'prob': tf.nn.softmax(logits)
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth, logits=logits)
    loss_mean = tf.reduce_mean(loss)

    #predicted_classes = tf.argmax(logits)

    accuracy = tf.metrics.accuracy(labels=ground_truth,
      predictions=logits,
      name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])


    # Create training ops.
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.1,
        use_nesterov=True,
        momentum=0.9)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_mean, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode = mode, loss=loss_mean, train_op=train_op)




    #creating test ops.
    eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(
      labels=tf.argmax(ground_truth,1), predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metric_ops)


print model(tf.estimator.ModeKeys.EVAL)
