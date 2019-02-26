import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from utils_libs import *
from bayesian_rnn import *

# ---- model restore and testing ----

def hyper_para_selection(hpara, 
                         error_log, 
                         val_epoch_num, 
                         test_epoch_num):
    
    val_err = []
    
    for hp_error in error_log:
        
        # based on RMSE
        val_err.append( mean([k[3] for k in hp_error[:val_epoch_num]]) )
    
    idx = val_err.index(min(val_err))
    
    return hpara[idx], [i[0] for i in error_log[idx]][:test_epoch_num], min(val_err)

def test_nn(epoch_samples, 
            x_test, 
            y_test, 
            file_path, 
            method_str, 
            dataset_str, 
            dropout_keep_prob, 
            bool_mc, 
            mc_n_samples, 
            bool_instance_eval,
            loss_type):
    
    # ensemble of model snapshots
    for idx in epoch_samples:
        
        # path of the stored models 
        tmp_meta = file_path + method_str + '_' + dataset_str + '_' + str(idx) + '.meta'
        tmp_data = file_path + method_str + '_' + dataset_str + '_' + str(idx)
        
        # clear graph
        tf.reset_default_graph()
        
        with tf.device('/device:GPU:0'):
            
            config = tf.ConfigProto()
        
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
        
            sess = tf.Session(config = config)
            
            if method_str == 'plain':
                
                reg = bayesian_rnn(sess, 
                                   loss_type = loss_type)
                
            else:
                print("[ERROR] RNN type ")
                return 
                
            # restore the model    
            reg.pre_train_restore_model(tmp_meta, tmp_data)
            
            # testing using the restored model
            rmse, mae, mape, nllk, py_mean, py_nllk, py_unc = reg.inference(x_test,
                                                                            y_test, 
                                                                            dropout_keep_prob if bool_mc == True else 1.0,
                                                                            bool_mc_dropout = bool_mc,
                                                                            n_samples = mc_n_samples,
                                                                            bool_instance = bool_instance_eval)
            
    return rmse, mae, mape, nllk, py_mean, py_nllk, py_unc
                          
                          
                          
                          
                          
                          
