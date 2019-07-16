"""
This is the tensorflow implementation of the NRML code. This is not complete yet .
Author: Md. Samiul Alam
Modified by: Md. Monowar Anjum
"""

import tensorflow as tf
import numpy as np


def dist_mat(P1, P2=None):
    if P2 is not None:
        X1 = tf.tile(tf.expand_dims(tf.reduce_sum(P1 ** 2, axis=-1), axis=-1), [1, tf.shape(P2)[0]])
        X2 = tf.tile(tf.expand_dims(tf.reduce_sum(P2 ** 2, axis=-1), axis=-1), [1, tf.shape(P1)[0]])

        R = P1 @ tf.transpose(P2)
        D = X1 + tf.transpose(X2) - 2 * R

        D = tf.sqrt(tf.cast(D, dtype=tf.complex64))
        return D
    else:
        X1 = tf.tile(tf.expand_dims(tf.reduce_sum(P1 ** 2, axis=-1), axis=-1), [1, tf.shape(P1)[0]])
        R = P1 @ tf.transpose(P1)
        D = X1 + tf.transpose(X1) - 2 * R

        D = tf.sqrt(tf.cast(D, dtype=tf.complex64))
        return D

class NRML:
    def __init__(self, N, K, d):
        self.N = N
        self.K = K
        self.convergence_data = []
        self.W = 0
        self.Wt = tf.Variable(np.zeros(shape=(d, d), dtype=np.float32))
        self.X = tf.placeholder(tf.float32, shape=[N, d])
        self.Y = tf.placeholder(tf.float32, shape=[N, d])

    def fit(self, X, Y, T=10, plot_convergence=False):
        deltas = []
        W = 0
        sess = tf.keras.backend.get_session()
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(T)):
            sess.run(self.step, {self.X: X, self.Y: Y})
            Wt = sess.run(self.Wt)
            deltas.append(np.mean((Wt - W) ** 2))
            X = X @ Wt
            Y = Y @ Wt
            W = Wt
        self.convergence_data = deltas
        self.W = W
        return (W, deltas)

    def build_model(self):
        X = self.X
        Y = self.Y
        with tf.variable_scope('nrml'):
            with tf.variable_scope('dist_H1'):
                H1 = self.__diff(X, Y) / (self.K * self.N)
            with tf.variable_scope('dist_H2'):
                H2 = self.__diff(Y, X) / (self.K * self.N)
            with tf.variable_scope('dist_H3'):
                D = X - Y
                H3 = (tf.transpose(D) @ D) / self.N

            H = (H1 + H2 - H3)

            e, Wt = tf.linalg.eigh(H)
            Wt = tf.reverse(Wt, axis=[-1])
            self.step = self.Wt.assign(Wt)
        return (X, Y, self.Wt)

    def __diff(self, x, y):
        D = dist_mat(y)
        I = tf.argsort(D)
        y_knn_idx = I[:, 1:self.K + 1]
        yt = tf.stack([tf.gather(y, i) for i in tf.unstack(y_knn_idx)], 1)
        x = tf.expand_dims(x, 0)
        d = x - yt
        return tf.reduce_sum(tf.transpose(d, perm=[0, 2, 1]) @ d, axis=0)

    def plot_convergence_curve(self):
        data = pd.DataFrame({'Iterations': range(len(self.convergence_data)),
                             '$\Delta$': self.convergence_data})
        sns.lineplot(x='Iterations',
                     y='$\Delta$',
                     data=data)