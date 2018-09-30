import tensorflow as tf

slim = tf.contrib.slim


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)


def network(net, scope, out_dim, is_training=True, reuse=False):
    with tf.variable_scope(scope, values=[net], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                            activation_fn=leaky_relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=tf.constant_initializer(0.0)):
            with slim.arg_scope([slim.conv2d], padding='VALID'):
                net = slim.stack(net, slim.conv2d, [(32, [8, 8], 4), (64, [4, 4], 2), (64, [3, 3], 1)], scope='conv')
                net = slim.flatten(net)
                net = slim.stack(net, slim.fully_connected, [512, out_dim], scope='fc')
                return net


def loss(dqn_out, target_qval, target_actions, scope):
    with tf.name_scope(scope):
        selected_qval = tf.gather_nd(dqn_out, target_actions)
        delta = target_qval - selected_qval
        loss = tf.reduce_mean(tf.where(
            tf.abs(delta) < 1., .5 * tf.square(delta), tf.abs(delta) - .5, name = 'huber_loss'))
    return loss



