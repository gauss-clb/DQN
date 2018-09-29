from collections import deque
import numpy as np
import random
from emulator import AtariEmulator
import nn
import tensorflow as tf
import Statistics
import os

slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(
    'train', True, 'Whether to train or test.')
tf.app.flags.DEFINE_boolean(
    'save', False, 'Whether to save the best result, save can be True only when train is False.')


tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'width', 84, 'The width of the states.')
tf.app.flags.DEFINE_integer(
    'height', 84, 'The height of the states.')
tf.app.flags.DEFINE_integer(
    'channels', 4, 'The channels of the states.')
tf.app.flags.DEFINE_integer(
    'numOfActs', 4, 'The number of actions.')
tf.app.flags.DEFINE_integer(
    'capacity', 100000, 'The capacity of replay memory.')
tf.app.flags.DEFINE_integer(
    'randomPlaySteps', 50000, 'The steps of randomly play.')
tf.app.flags.DEFINE_integer(
    'epochs', 200, 'The epochs of training process.')
tf.app.flags.DEFINE_integer(
    'stepsPerEpoch', 250000, 'The steps of per epoch.')
tf.app.flags.DEFINE_integer(
    'stepsPerCopy', 10000, 'The steps of per copy.')
tf.app.flags.DEFINE_integer(
    'learnInterval', 4, 'The steps of per updating parameter.')
tf.app.flags.DEFINE_integer(
    'epsilonStep', 1000000, 'The steps of epsilon decay.')
tf.app.flags.DEFINE_integer(
    'ckpt_version', 195, 'The version of checkpoint file to load.')



tf.app.flags.DEFINE_float(
    'epsilon', 1., 'The epsilon of epsilon greedy.')
tf.app.flags.DEFINE_float(
    'epsilonStart', 1., 'The start of epsilon decay.')
tf.app.flags.DEFINE_float(
    'epsilonEnd', .1, 'The end of epsilon decay.')
tf.app.flags.DEFINE_float(
    'gamma', .99, 'The discount of reward.')


tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'checkpoint',
    'The directory to save or load checkpoint file.')


FLAGS = tf.app.flags.FLAGS


def assignList(target_scope, original_scope):
    target_list = slim.get_model_variables(scope=target_scope)
    orignal_list = slim.get_model_variables(scope=original_scope)
    as_li = []
    for tar, ori in zip(target_list, orignal_list):
        assert tar.op.name[tar.op.name.find('/'):] == ori.op.name[ori.op.name.find('/'):], 'Parameters are mismatched!'
        as_li.append(tar.assign(ori))
    return as_li


def epsilonGreedy(s_t, dqn_out, sess, epsilon = .05):
    if np.random.rand() < epsilon:
        return np.random.randint(0, FLAGS.numOfActs)
    else:
        qval = dqn_out.eval(feed_dict = {state: s_t[np.newaxis] / 255.}, session = sess)
        return np.argmax(qval[0])


def oneStep(s_t, dqn_out, sess, epsilon, replay_mem, statistics = None):
    a_t = epsilonGreedy(s_t, dqn_out, sess, epsilon)
    s_t_plus_1, r_t, isTerminal = ale.next(a_t)
    s_t_copy = s_t.copy()
    s_t[..., : FLAGS.channels-1] = s_t[..., 1:]
    s_t[..., -1] = s_t_plus_1
    s_t_plus_1_copy = s_t.copy()
    replay_mem.append((s_t_copy, a_t, r_t, s_t_plus_1_copy))
    if statistics:
        statistics.statistics(a_t, r_t, isTerminal, epsilon)
    return isTerminal


def sampleBatch(replay_mem):
    mini_batch = random.sample(replay_mem, FLAGS.batch_size)
    s_t = np.zeros((FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.channels), dtype=np.uint8)
    a_t = np.zeros((FLAGS.batch_size), dtype=np.int32)
    r_t = np.zeros((FLAGS.batch_size), dtype=np.int32)
    s_t_plus_1 = np.zeros((FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.channels), dtype=np.uint8)
    for i in range(FLAGS.batch_size):
        s_t[i], a_t[i], r_t[i], s_t_plus_1[i] = mini_batch[i]
    return s_t, a_t, r_t, s_t_plus_1


def deepQLearning(target_out, sess, replay_mem):
    s_t, a_t, r_t, s_t_plus_1 = sampleBatch(replay_mem)
    _target_out = target_out.eval(feed_dict={state: s_t_plus_1 / 255.}, session = sess)
    _target_actions = np.stack([np.arange(FLAGS.batch_size), a_t], axis=-1)
    _target_qval = r_t + FLAGS.gamma * np.max(_target_out, axis=-1)
    _loss, _ = sess.run([loss, rmsprop], feed_dict = {state: s_t / 255., target_qval: _target_qval, target_actions: _target_actions})
    return _loss


if __name__  == '__main__':

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    
    ale = AtariEmulator(b'breakout.bin', not FLAGS.train, FLAGS.save)
    if FLAGS.train:
        statistics = Statistics.Statistics()
    else:
        statistics = Statistics.Statistics('test.csv')
    state = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, FLAGS.channels])
    target_qval = tf.placeholder(tf.float32, [None])
    target_actions = tf.placeholder(tf.int32, [None, 2])
    
    # build network
    print('Start to build QNet and Target QNet ...')
    dqn_out = nn.network(state, 'dqn', FLAGS.channels)
    target_out = nn.network(state, 'target', FLAGS.channels)
    assign_li = assignList('target', 'dqn')
    loss = nn.loss(dqn_out, target_qval, target_actions, scope = 'dqn/loss')

    with tf.name_scope('dqn/optimizer'):
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.maximum(0.00025, tf.train.exponential_decay(
            0.00025, global_step, 50000, 0.96, staircase = True))
        opt = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.95, epsilon = 0.01)
        rmsprop = opt.minimize(loss, global_step = global_step)

    replay_mem = deque(maxlen=FLAGS.capacity)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    saver = tf.train.Saver(slim.get_model_variables(scope='dqn'))

    test_s_t = np.zeros([FLAGS.height, FLAGS.width, FLAGS.channels], dtype=np.uint8)

    if FLAGS.train:
        # randomly play
        print('Start to randomly play ...')
        s_t = np.zeros([FLAGS.height, FLAGS.width, FLAGS.channels], dtype=np.uint8)
        ale.setMode('train')
        ale.randomStart(s_t)
        for i in range(FLAGS.randomPlaySteps):
            isTerminal = oneStep(s_t, dqn_out, sess, 1., replay_mem)
            if isTerminal:
                ale.start()

        steps = 0
        epsilon = FLAGS.epsilon
        epsilonGap = FLAGS.epsilonStart - FLAGS.epsilonEnd

        import time
        for i in range(FLAGS.epochs):
            # training 
            print('Start to train ...')
            ale.setMode('train')
            statistics.reset()
            for j in range(FLAGS.stepsPerEpoch):
                # copy parameters
                if j % FLAGS.stepsPerCopy == 0:
                    sess.run(assign_li)
                isTerminal = oneStep(s_t, dqn_out, sess, epsilon, replay_mem, statistics)
                if j % FLAGS.learnInterval == 0:
                    _loss = deepQLearning(target_out, sess, replay_mem)
                    statistics.lossStatistics(_loss)
                if isTerminal:
                    ale.randomStart(s_t)
                steps += 1
                epsilon = max(.1, 1. - steps * epsilonGap / FLAGS.epsilonStep)
            statistics.write(i + 1, 'train')
            if (i + 1) % 3 == 0:
                path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, path, global_step = i + 1)
            
            # testing
            print('Start to test ...')
            ale.setMode('test')
            statistics.reset()
            for j in range(FLAGS.stepsPerEpoch // 2):
                isTerminal = oneStep(test_s_t, dqn_out, sess, .05, replay_mem, statistics)
                if isTerminal:
                    ale.randomStart(test_s_t)
            statistics.write(i + 1, 'test')
    else:
        saver.restore(sess, 'checkpoint/model.ckpt-%d' % FLAGS.ckpt_version)
        isTerminal = True
        for i in range(10):
            ale.setMode('test')
            statistics.reset()
            for j in range(FLAGS.stepsPerEpoch // 20):
                isTerminal = oneStep(test_s_t, dqn_out, sess, .05, replay_mem, statistics)
                if isTerminal:
                    ale.randomStart(test_s_t)
            statistics.write(i + 1, 'test')



