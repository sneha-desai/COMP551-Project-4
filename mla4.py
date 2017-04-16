# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:43:24 2017

@author: Riyad Imrul
"""

import os,sys
import numpy as np
#from generate_synthetic_data import *
sys.path.insert(0, '../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
loss_function = lf._logistic_loss

apply_fairness_constraints = 1
apply_accuracy_constraint = 0
sep_constraint = 0
gamma = 0
sensitive_attrs = ["s1"]
sensitive_attrs_to_cov_thresh ={"s1":0}


x=np.load('x2.npy')

#x_train=x_train[0:-1]
x_control=np.load('x_control2.npy')

x_control={'s1':x_control}

y=np.load('y2.npy')
y=-2*y+1                # converting 1 0 values to +1 -1; and reversing the connotation

ut.compute_p_rule(x_control["s1"], y)


x = ut.add_intercept(x) # add intercept to X before applying the linear classifier
train_fold_size = 0.8
x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(x, y, x_control, train_fold_size)


w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

#num_folds=2
#w=ut.compute_cross_validation_error(x_train, y_train, x_control_train, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None)
#x_test=x
#y_test=y
#x_control_test=x_control

train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
distances_boundary_test = (np.dot(x_test, w)).tolist()
all_class_labels_assigned_test = np.sign(distances_boundary_test)
correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	

print('After optimization: P',p_rule)
print('Accuracy',test_score)
#distance_boundary = np.dot(w, x_train) # will give the distance from the decision boundary

