#!/usr/bin/python
import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

import scipy as sp

# local 
from utils_libs import *
from rnn_attention import *
from rnn_layers import *

# ---- Bayesian RNN ----

class bayesian_rnn():
    
    def __init__(self, session, loss_type):
        
        self.LEARNING_RATE = 0.0
        self.L2 = 0.0
        
        self.n_lstm_dim_layers = 0.0
        
        self.N_STEPS = 0.0
        self.N_DATA_DIM = 0.0
        
        self.att_type = 0.0
        self.loss_type = loss_type
        
        # placeholders
        self.x = 0.0
        self.y = 0.0
        self.keep_prob = 0.0
        
        self.sess = session
        
    
    def network_ini(self, 
                    n_lstm_dim_layers, 
                    n_steps, 
                    n_data_dim,
                    lr, 
                    l2_dense, 
                    max_norm, 
                    bool_residual, 
                    att_type, 
                    l2_att, 
                    num_dense,
                    bool_regular_attention, 
                    bool_regular_lstm, 
                    bool_regular_dropout_output, 
                    lr_decay,
                    loss_type):
        
        # ---- fix the random seed to reproduce the results
        np.random.seed(1)
        tf.set_random_seed(1)
        
        # ---- ini
        
        self.l2 =  l2_dense
        
        self.n_lstm_dim_layers = n_lstm_dim_layers
        
        self.N_STEPS    = n_steps
        self.N_DATA_DIM = n_data_dim
        
        self.att_type = att_type
        
        if lr_decay == True:
            self.lr = tf.Variable(lr, trainable = False)
        else:
            self.lr = lr
            
        self.new_lr = tf.placeholder(tf.float32, shape = (), name = 'new_lr')
        
        self.loss_type = loss_type
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.N_STEPS, self.N_DATA_DIM], name = 'x')
        self.y = tf.placeholder(tf.float32, [None, 1], name = 'y')
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = 'keep_prob')
        
        # ---- network architecture 
        
        h = self.x
        
        # dropout on input?
        h = tf.nn.dropout(h, self.keep_prob)
        
        # dropout
        h, _ = plain_lstm(h, 
                          n_lstm_dim_layers, 
                          'lstm', 
                          self.keep_prob)
        
        # attention
        if att_type == 'temp':
            
            print('\n --- Plain RNN using temporal attention:\n')
            
            # dropout ?
            h, self.att, regu_att = attention_temp_logit(h, 
                                                         n_lstm_dim_layers[-1], 
                                                         'att', 
                                                         self.N_STEPS)
            
            # dropout
            h, regu_multi_dense, out_dim = multi_dense(h, 
                                                       2*n_lstm_dim_layers[-1], 
                                                       num_dense, 
                                                       'multi_dense', 
                                                       self.keep_prob, 
                                                       max_norm)
            
        else:
            
            print('\n --- Plain RNN using NO attention:\n')
            
            # obtain the last hidden state
            tmp_hiddens = tf.transpose(h, [1,0,2])
            h = tmp_hiddens[-1]
            
            # dropout
            h, regu_multi_dense, out_dim = multi_dense(h, 
                                                       n_lstm_dim_layers[-1], 
                                                       num_dense, 
                                                       'multi_dense', 
                                                       self.keep_prob, 
                                                       max_norm)
            
        self.regularization = l2_dense*regu_multi_dense
        
        
        # ---- output layer
        # remove the constraints of max_norm regularization 
        
        # dropout before the output layer
        if bool_regular_dropout_output == True:
            h = tf.nn.dropout(h, self.keep_prob)
            
        # ?
        self.mean, self.regu_mean = dense(x = h, 
                                     x_dim = out_dim, 
                                     out_dim = 1, 
                                     scope = "output_mean", 
                                     dropout_keep_prob = 1.0,
                                     max_norm_regul = 0.0,
                                     bool_no_activation = True)
        
        # ?
        tmp_var, self.regu_var = dense(x = h,
                                  x_dim = out_dim, 
                                  out_dim = 1, 
                                  scope = "output_var", 
                                  dropout_keep_prob = 1.0,
                                  max_norm_regul = 0.0,
                                  bool_no_activation = True)
        
        '''        
        h, regu_dense = dense(x = h, 
                              x_dim = out_dim, 
                              out_dim = 2, 
                              scope = "output_dense", 
                              dropout_keep_prob = 1.0, 
                              max_norm_regul = max_norm,
                              bool_no_activation = True)
        '''
        
        # [B 1] [B 1]
        #self.mean, tmp_var = tf.split(h, [1, 1], 1)   
        
        self.var = tf.square(tmp_var)
        self.inv_var = tf.square(tmp_var)
                
        #self.regularization += l2_dense*regu_dense
        
        
        # ---- log likelihood 
        
        # llk: log likelihood 
        # nllk: negative log likelihood
        # py: predicted y
        # inv: inversed variance
        
        # [B 1]
        #tmp_lk = tf.exp(-0.5*tf.square(self.y - self.mean)/(self.var + 1e-5))/(2.0*np.pi*(self.var + 1e-5))**0.5
        #self.neg_llk = tf.reduce_sum(-1.0*tf.log(tmp_lk + 1e-5))
        
        #self.py_nllk = 0.5*tf.square(self.y - self.mean)/(self.var + 1e-5) + 0.5*tf.log(self.var + 1e-5)
        #self.nllk = tf.reduce_sum(self.py_nllk)
        
        # numerical stable
        
        # [B 1]
        #tmp_lk_inv = tf.exp(-0.5*tf.square(self.y - self.mean)*self.inv_var)/(2.0*np.pi)**0.5*(self.inv_var**0.5)
        #self.neg_llk_inv = tf.reduce_sum(-1.0*tf.log(tmp_lk_inv + 1e-5))
        
        self.py_nllk_inv = 0.5*tf.square(self.y - self.mean)*self.inv_var - 0.5*tf.log(self.inv_var + 1e-5)
        self.nllk_inv = tf.reduce_sum(self.py_nllk_inv)
        
        # [B 1]
        #tmp_lk_pseudo = tf.exp(-0.5*tf.square(self.y - self.mean))/(2.0*np.pi)**0.5
        #self.pseudo_neg_llk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(pseudo_lk, 1)+1e-5))
        
        self.py_nllk_pseudo = 0.5*tf.square(self.y - self.mean)
        self.nllk_pseudo = tf.reduce_sum(self.py_nllk_pseudo)
            
        # ---- regularization
        
        if bool_regular_attention == True:
            
            self.regularization += 0.1*l2_dense*(regu_att)
        
        if bool_regular_lstm == True:
            
            self.regul_lstm = sum(tf.nn.l2_loss(tf_vari) for tf_vari in tf.trainable_variables() if ("lstm" in tf_vari.name))
            
            self.regularization += 0.1*l2_dense*self.regul_lstm
    
    
    def train_ini(self):
        
        self.square_error = tf.reduce_sum(tf.square(self.y - self.mean))
        
        # unc: uncertainty, standard deviation
        
        # loss function 
        if self.loss_type == 'mse':
            
            self.mse = tf.reduce_mean(tf.square(self.y - self.mean))
            
            self.loss = self.mse + self.regularization + self.l2*self.regu_mean
            
            self.neg_llk = self.nllk_pseudo
            
            self.py_neg_llk = self.py_nllk_pseudo
            self.py_mean = self.mean
            self.py_unc = 0.0
        
        # numerical stable
        elif self.loss_type == 'lk_inv':
            
            self.loss = self.nllk_inv + self.regularization + self.l2*(self.regu_mean + self.regu_var)
            
            self.neg_llk = self.nllk_inv
            
            self.py_neg_llk = self.py_nllk_inv
            self.py_mean = self.mean
            self.py_unc = tf.sqrt(1.0/(self.inv_var + 1e-5))
            
        else:
            print('--- [ERROR] loss type')
            return
            
        '''    

        elif self.loss_type == 'lk':
            
            self.loss = self.neg_llk + self.regularization
            
            self.neg_llk = self.nllk
            
            self.py_neg_llk = self.py_nllk_inv
            self.py_mean = self.mean
            self.py_unc = np.sqrt(self.var)
            
        
        elif self.loss_type == 'lk_pseudo':
            
            self.loss = self.neg_llk_pseudo + self.regularization
            
            self.neg_llk = self.nllk_pseudo
            
            self.py_neg_llk = self.py_nllk_pseudo
            self.py_mean = self.mean
            self.py_unc = 0.0
        '''
        
        # optimizer 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)  
        
        # initilization 
        self.init = tf.global_variables_initializer()
        self.sess.run( [self.init] )
        
#   initialize inference         
    def inference_ini(self):
        
        # error metrics
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.mean)))
        self.mae = tf.reduce_mean(tf.abs(self.y - self.mean))
        
        # filtering before mape calculation
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.mean, mask)
        
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)*1.0/(y_mask+1e-10)))
        
        # predicting by pre-trained models
        tf.add_to_collection("rmse", self.rmse)
        tf.add_to_collection("mae", self.mae)
        tf.add_to_collection("mape", self.mape)
        
        tf.add_to_collection("neg_llk", self.neg_llk)
        
        tf.add_to_collection("py_neg_llk", self.py_neg_llk)
        tf.add_to_collection("py_mean", self.py_mean)
        tf.add_to_collection("py_unc", self.py_unc)
        
    def train_batch(self, 
                    x_batch, 
                    y_batch, 
                    keep_prob, 
                    bool_lr_update, 
                    lr):
        
        # learning rate decay update
        if bool_lr_update == True:
            
            lr_update = tf.assign(self.lr, self.new_lr)
            _ = self.sess.run([lr_update], feed_dict={self.new_lr:lr})
        
        _, tmp_loss, tmp_sqsum = self.sess.run([self.optimizer, self.loss, self.square_error],\
                                                feed_dict = {self.x:x_batch, 
                                                             self.y:y_batch, 
                                                             self.keep_prob:keep_prob})
        return tmp_loss, tmp_sqsum
    
    
#   inference givn testing data
    def inference(self, 
                  x_test, 
                  y_test, 
                  keep_prob,
                  bool_mc_dropout,
                  n_samples, 
                  bool_instance):
        
        print("??? test ????", keep_prob, bool_mc_dropout, n_samples)
        
        if bool_mc_dropout == True:
            
            # [n_sample, B, 1]
            samples_py_mean = []
            samples_py_nllk = []
            samples_py_unc  = []
                
            for i in range(n_samples):
                
                # randomness
                #tf.set_random_seed(i)
                
                if self.loss_type == 'mse':
                    
                    tmp_py_mean, tmp_py_nllk = self.sess.run([tf.get_collection('py_mean')[0],
                                                              tf.get_collection('py_neg_llk')[0]],
                                                       feed_dict = {'x:0':x_test, 'y:0':y_test, 'keep_prob:0':keep_prob})
                    samples_py_mean.append(tmp_py_mean)
                    samples_py_nllk.append(tmp_py_nllk)
                
                elif 'lk' in self.loss_type:
                    
                    tmp_py_mean, tmp_py_nllk, tmp_py_unc = self.sess.run([tf.get_collection('py_mean')[0],
                                                                          tf.get_collection('py_neg_llk')[0],
                                                                          tf.get_collection('py_unc')[0]],
                                                       feed_dict = {'x:0':x_test, 'y:0':y_test, 'keep_prob:0':keep_prob})
                    samples_py_mean.append(tmp_py_mean)
                    samples_py_nllk.append(tmp_py_nllk)
                    samples_py_unc.append(tmp_py_unc)
            
            # ? test
            sample_subset = np.transpose(np.asarray(samples_py_mean), [1, 0, 2])[0][:10]
            
            print("??? test ????", sample_subset, y_test[0])
            
            py_mean = np.mean(np.transpose(np.asarray(samples_py_mean), [1, 0, 2]), axis = 1)
            
            rmse = self.metric_rmse(y = y_test, py = py_mean)
            mae = self.metric_mae(y = y_test, py = py_mean)
            mape = self.metric_mape(y = y_test, py = py_mean)
            
            py_nllk = self.mc_neg_llk(np.transpose(np.asarray(samples_py_nllk), [1, 0, 2]), n_samples)
            nllk = np.sum(py_nllk)
            
        else:
            
            
            py_mean, rmse, mae, mape, nllk = self.sess.run([tf.get_collection('py_mean')[0],
                                                            tf.get_collection('rmse')[0], 
                                                   tf.get_collection('mae')[0],
                                                   tf.get_collection('mape')[0],
                                                   tf.get_collection('neg_llk')[0]],
                                                   feed_dict = {'x:0':x_test, 'y:0':y_test, 'keep_prob:0':keep_prob})
            
            mape = self.metric_mape(y = y_test, py = py_mean)
        
        
        # return [rmse, mae, mape, nllk, py_mean, py_unc, py_nllk] 
        if bool_instance == True:
            
            
            if bool_mc_dropout == True:
                
                # [n_sample, B, 1]
                
                # py_mean
                py_mean = np.mean(np.transpose(np.asarray(samples_py_mean), [1, 0, 2]), axis = 1)
                
                # py_nllk
                # py_nllk = self.mc_neg_llk(np.transpose(np.asarray(samples_py_nllk), [1, 0, 2]), n_samples)
                
                # py_unc
                if self.loss_type == 'mse':
                    
                    py_unc = np.std(np.transpose(np.asarray(samples_py_mean), [1, 0, 2]), axis = 1)
                
                elif 'lk' in self.loss_type:
                    
                    # [B n_sample 1]
                    tmp_sample_var = np.square(np.transpose(np.asarray(samples_py_unc)**2, [1, 0, 2]))
                    tmp_sample_mean = np.transpose(np.asarray(samples_py_mean), [1, 0, 2])
                    
                    # standard deviation 
                    py_unc = np.sqrt(np.mean(tmp_sample_var + tmp_sample_mean**2, axis = 1) - py_mean**2)
                
            else:
                
                if self.loss_type == 'mse':
                    
                    py_mean, py_nllk = self.sess.run([tf.get_collection('py_mean')[0],
                                                      tf.get_collection('py_neg_llk')[0]], 
                                            feed_dict = {'x:0':x_test, 'y:0':y_test, 'keep_prob:0':keep_prob})
                    py_unc = None
                
                elif 'lk' in self.loss_type:
                    
                    py_mean, py_nllk, py_unc = self.sess.run([tf.get_collection('py_mean')[0], 
                                                              tf.get_collection('py_neg_llk')[0], 
                                                              tf.get_collection('py_unc')[0]], 
                                                       feed_dict = {'x:0':x_test, 'y:0':y_test, 'keep_prob:0':keep_prob})
                    
        # only return [rmse, mas, mape, nllk]   
        else:

            py_mean = None
            py_nllk = None
            py_unc = None
            
        return rmse, mae, mape, nllk, py_mean, py_nllk, py_unc
            
    
    def mc_neg_llk(self, py_nllk, n_samples):
        
        # numerical stable 
        # [B n_sample, 1]
        return -1*sp.special.logsumexp(-1.0*py_nllk, axis = 1) + np.log(n_samples)
    
    def metric_rmse(self, y, py):
        return sqrt(np.mean((y - py)**2))
    
    def metric_mae(self, y, py):
        return np.mean(abs(y - py))
    
    def metric_mape(self, y, py):
        
        tmp_idx = []
        for idx, tmp in enumerate(y):
            if abs(tmp)>1e-5:
                tmp_idx.append(idx)
        
        return np.mean(1.0*abs(y[tmp_idx] - py[tmp_idx])/(y[tmp_idx]+1e-10))
                         
    
    # restore the model from the files
    def pre_train_restore_model(self, 
                                path_meta, 
                                path_data):
        
        saver = tf.train.import_meta_graph(path_meta, clear_devices=True)
        saver.restore(self.sess, path_data)
        
        return 0
    
    '''
    # inference using pre-trained model 
    def pre_train_inference(self,
                            x_test,
                            y_test,
                            keep_prob,
                            bool_mc_dropout,
                            n_samples):
        
        if bool_mc_dropout == True:
            
            # [n_sample, B, 1]
            samples = []
            for i in n_samples:
                samples.append(self.sess.run([tf.get_collection('pred')[0]], 
                                              feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob}))
                
            py = np.mean(np.transpose(np.asarray(samples), [1, 0, 2]), axis = 1)
            return py, self.rmse(py, y_test), self.mae(py, y_test), self.mape(py, y_test) 
        
        else:
            return self.sess.run([tf.get_collection('pred')[0],
                                  tf.get_collection('rmse')[0],
                                  tf.get_collection('mae')[0],
                                  tf.get_collection('mape')[0]],
                                 feed_dict = {'x:0': x_test, 'y:0': y_test, 'keep_prob:0': keep_prob})
   '''