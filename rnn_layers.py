import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from utils_libs import *
#from mv_rnn_cell import *
#from ts_mv_rnn import *



# ---- residual and plain dense layers ----  
    
def res_lstm(x, hidden_dim, n_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
            #Deep lstm: residual or highway connections 
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1, n_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            tmp_h = hiddens
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
            hiddens = hiddens + tmp_h 
             
    return hiddens, state

def plain_lstm(x, dim_layers, scope, dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
        
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0], \
                                               initializer= tf.contrib.keras.initializers.glorot_normal())
            
            # dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell, state_keep_prob = dropout_keep_prob)
            
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
        
        
    for i in range(1,len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[i], \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            
            # dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell, state_keep_prob = dropout_keep_prob)
            
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
                
    return hiddens, state 

    
def res_dense(x, x_dim, hidden_dim, n_layers, scope, dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, hidden_dim], dtype = tf.float32,
                                         initializer = tf.contrib.layers.variance_scaling_initializer())
                                         #initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([hidden_dim]))
                h = tf.nn.relu(tf.matmul(x, w) + b )

                regularization = tf.nn.l2_loss(w)
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, n_layers):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [hidden_dim, hidden_dim], \
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( hidden_dim ))
                
                # residual connection
                tmp_h = h
                h = tf.nn.relu( tf.matmul(h, w) + b )
                h = tmp_h + h
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization
    
def plain_dense(x, x_dim, dim_layers, scope, dropout_keep_prob, max_norm_regul):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # max norm constraints
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(x, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(x, w) + b )
                    
                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        # dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( dim_layers[i] ))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(h, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization
    
    
def multi_dense(x, x_dim, num_layers, scope, dropout_keep_prob, max_norm_regul, activation_type):
    
        in_dim = x_dim
        out_dim = int(in_dim/2)
        
        h = x
        regularization = 0.0
        
        for i in range(num_layers):
            
            with tf.variable_scope(scope + str(i)):
                
                # dropout
                h = tf.nn.dropout(h, dropout_keep_prob)
                
                w = tf.get_variable('w', 
                                    [in_dim, out_dim], 
                                    dtype=tf.float32,\
                                    initializer = tf.contrib.layers.xavier_initializer())
                                    #tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(out_dim))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    tmp_h = tf.matmul(h, clip_w) + b
                    regularization += tf.nn.l2_loss(clip_w)
                    
                    #h = tf.nn.relu(tf.matmul(h, clip_w) + b)
                    
                else:
                    
                    tmp_h = tf.matmul(h, w) + b
                    regularization += tf.nn.l2_loss(w)
                    
                
                # nonlinear activation 
                if activation_type == 'relu':
                    
                    h = tf.nn.relu(tmp_h)
                    
                elif activation_type == 'leaky_relu':
                    
                    # leaky relu
                    h = tf.maximum(tmp_h, 0.3*tmp_h)
                    
                else:
                    print("\n [ERROR] activation type, multi-dense \n")
                
                
                #?
                # regularization += tf.nn.l2_loss(w)
                # regularization += tf.reduce_sum(tf.abs(w))
                
                in_dim = out_dim
                out_dim = int(out_dim/2)
                
        return h, regularization, in_dim    
    
def dense(x, x_dim, out_dim, scope, dropout_keep_prob, max_norm_regul, activation_type):
    
    h = x
    regularization = 0.0
    
    with tf.variable_scope(scope):
        
        # dropout on the input
        h = tf.nn.dropout(h, dropout_keep_prob)
        w = tf.get_variable('w', 
                            [x_dim, out_dim], 
                            dtype = tf.float32,\
                            initializer = tf.contrib.layers.xavier_initializer())
                                    #variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(out_dim))
                
        # max norm constraints 
        if max_norm_regul > 0:
            
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
            clip_w = tf.assign(w, clipped)
                    
            tmp_h = tf.matmul(h, clip_w) + b
            regularization = tf.nn.l2_loss(clip_w)
                    
        else:
            
            tmp_h = tf.matmul(h, w) + b
            regularization = tf.nn.l2_loss(w)
        
        
        # activation 
        if activation_type == 'relu':
            h = tf.nn.relu(tmp_h)
                    
        elif activation_type == 'leaky_relu':
            # leaky relu
            h = tf.maximum(tmp_h, 0.3*tmp_h)
            
        elif activation_type == '':
            h = tmp_h
            
        else:
            print("\n [ERROR] activation type, dense \n")
            
        '''
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu(tmp_h) 
        '''
        #?
        #regularization = tf.nn.l2_loss(w)
                
    return h, regularization



