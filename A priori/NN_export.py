# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:19:45 2023

@author: P27
"""


#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


directory = 'NN_24'


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph






tf.keras.backend.clear_session()
direct = directory + '/model.h5'
model = tf.keras.models.load_model(direct,compile=False)


save_directory = directory + '/'
save_direct = 'model.pb'
frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, save_directory, save_direct, as_text=False)






#%%

import h5py
import numpy as np

filename = directory + '/scaling_parameters.h5'
scaling_params = h5py.File(filename, 'r')

scaling_params.keys()




