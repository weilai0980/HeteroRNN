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

# ---- model restore and testing ----

def hyper_para_selection(hpara, error_log, top_k):
    
    val_err = []
    
    for hp_error in error_log:
        
        # based on RMSE
        val_err.append( mean([k[3] for k in hp_error[:top_k]]) )
    
    idx = val_err.index(min(val_err))
    
    return hpara[idx], [i[0] for i in error_log[idx]][:top_k], min(val_err)

def test_nn(epoch_samples, x_test, y_test, file_path, method_str):
    
    for idx in epoch_samples:
        
        tmp_meta = file_path + method_str + '-' + str(idx) + '.meta'
        tmp_data = file_path + method_str + '-' + str(idx)
        
        # clear graph
        tf.reset_default_graph()
        
        with tf.device('/device:GPU:0'):
            
            config = tf.ConfigProto()
        
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
        
            sess = tf.Session(config = config)
            
            if method_str == 'plain':
                
                # restore the model
                reg = tsLSTM_plain(sess)
                
            elif method_str == 'mv_tensor' or method_str == 'mv_full':
                
                reg = tsLSTM_mv(sess)
                      
                
            reg.pre_train_restore_model(tmp_meta, tmp_data)
            # testing using the restored model
            yh, rmse, mae, mape = reg.pre_train_inference(x_test, y_test, 1.0)
                
    return yh, rmse, mae, mape


# ---- mv dense layers ---- 

def multi_mv_dense(num_layers, keep_prob, h_vari, dim_vari, scope, num_vari, \
                   bool_no_activation, max_norm_regul, regul_type):
    
    in_dim_vari = dim_vari
    out_dim_vari = int(dim_vari/2)
    h_mv_input = h_vari
    
    reg_mv_dense = 0.0
    
    for i in range(num_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            # ? dropout
            h_mv_input = tf.nn.dropout(h_mv_input, keep_prob)
            # h_mv [V B d]
            # ? max norm constrains
            h_mv_input, tmp_regu_dense = mv_dense(h_mv_input, 
                                                  in_dim_vari,
                                                  scope + str(i),
                                                  num_vari, 
                                                  out_dim_vari,
                                                  False, 
                                                  max_norm_regul, 
                                                  regul_type)
            
            reg_mv_dense += tmp_regu_dense
            
            in_dim_vari  = out_dim_vari
            out_dim_vari = int(out_dim_vari/2)
            
    return h_mv_input, reg_mv_dense, in_dim_vari
            

# with max-norm regularization 
def mv_dense(h_vari, dim_vari, scope, num_vari, dim_to, bool_no_activation, max_norm_regul, regul_type):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w', [ num_vari, 1, dim_vari, dim_to ], initializer=tf.contrib.layers.xavier_initializer())
        # [V 1 1 d]
        b = tf.Variable( tf.random_normal([ num_vari, 1, 1, dim_to ]) )
        
        # [V B D 1]
        h_expand = tf.expand_dims(h_vari, -1)
        
        # max-norm regularization 
        if max_norm_regul > 0:
            
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 2)
            clip_w = tf.assign(w, clipped)
            
            tmp_h =  tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu(tmp_h) 
        
        # regularization type
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum( tf.abs(w) ) 
        
        else:
            return '[ERROR] regularization type'

# with max-norm regularization 
def mv_dense_share(h_vari, dim_vari, scope, num_vari, dim_to, bool_no_activation, max_norm_regul, regul_type):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [D d]
        w = tf.get_variable('w', [ dim_vari, dim_to ], initializer=tf.contrib.layers.xavier_initializer())
        # [ d]
        b = tf.Variable( tf.random_normal([ dim_to ]) )
        
        
        if max_norm_regul > 0:
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 0)
            clip_w = tf.assign(w, clipped)
            
            # [V B d]
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h = tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu( tmp_h ) 
            
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum( tf.abs(w) ) 
        
        else:
            return '[ERROR] regularization type'


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

def plain_dense_leaky(x, x_dim, dim_layers, scope, dropout_keep_prob, alpha):
    
        # dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # ?
                tmp_h = tf.matmul(x, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )

                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        # dropout
        # h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(dim_layers[i]))
                
                # ?
                tmp_h = tf.matmul(h, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization
    
    
def multi_dense(x, x_dim, num_layers, scope, dropout_keep_prob, max_norm_regul):
    
        in_dim = x_dim
        out_dim = int(in_dim/2)
        
        h = x
        regularization = 0.0
        
        for i in range(num_layers):
            
            with tf.variable_scope(scope+str(i)):
                
                # dropout
                h = tf.nn.dropout(h, dropout_keep_prob)
                
                w = tf.get_variable('w', [ in_dim, out_dim ], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( out_dim ))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(h, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                # regularization += tf.reduce_sum(tf.abs(w))
                
                in_dim = out_dim
                out_dim = int(out_dim/2)
                
        return h, regularization, in_dim    
    
def dense(x, x_dim, out_dim, scope, dropout_keep_prob, max_norm_regul, bool_no_activation):
    
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
                    
        else:
            tmp_h = tf.matmul(h, w) + b
        

        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu(tmp_h) 
        
        #?
        regularization = tf.nn.l2_loss(w)
                
    return h, regularization



