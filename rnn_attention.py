import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
#from mv_rnn_cell import *
from utils_libs import *

    
# ---- Attention plain ----

# ref: a structured self attentive sentence embedding  
def attention_temp_mlp( h, h_dim, att_dim, scope ):
    # tf.tensordot
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, att_dim], initializer=tf.contrib.layers.xavier_initializer())
        #? bias and nonlinear activiation 
        tmp_h = tf.nn.relu( tf.tensordot(h, w, axes=1) )

        w_logit = tf.get_variable('w_log', [att_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.tensordot(tmp_h, w_logit, axes=1)
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
    return tf.reduce_sum(h*tf.expand_dims(alphas, -1), 1), alphas


def attention_temp_logit( h, h_dim, scope, step ):
    # tf.tensordot
    
    h_context, h_last = tf.split(h, [step-1, 1], 1)
    h_last = tf.squeeze(h_last)
    
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', [h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([1, 1]))
        
        #? bias and nonlinear activiation 
        logit = tf.squeeze( tf.nn.tanh(tf.tensordot(h_context, w, axes=1) + b) )
        
        alphas = tf.nn.softmax( tf.squeeze(logit) )
        
        context = tf.reduce_sum(h_context*tf.expand_dims(alphas, -1), 1)
        
    return tf.concat([context, h_last], 1), alphas, tf.nn.l2_loss(w) 
    