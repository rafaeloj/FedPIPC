import flwr as fl
import tensorflow as tf
from DatasetManager import DatasetManager
from ModelManager import ModelManager
import numpy as np
import pickle
import os
import typing as T
import pandas as pd

import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class V2LetsFLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, model_type, dataset_name, n_partitions):
        self.cid: int = cid
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.n_partitions = n_partitions
        self.last_participation = None
        self.maxfl_w = 0

        print(f"Client {self.cid} initialized")

        # --- Initialize Dataset ---
        self.dm = DatasetManager(
            dataset_name,
            n_partitions,
            non_iid=True,
            cid = cid,
        )
    
        # --- Initialize model ---
        self.tail, self.head, self.model = ModelManager(model_type).get_model(
            input_shape=self.dm.X_train.shape[1:],
            n_classes=self.dm.n_classes
        )
        self.glob_tail, self.glob_head, self.glob_model = ModelManager(model_type).get_model(
            input_shape=self.dm.X_train.shape[1:],
            n_classes=self.dm.n_classes
        )
        self.tmp_tail, self.tmp_head, self.tmp_model = ModelManager(model_type).get_model(
            input_shape=self.dm.X_train.shape[1:],
            n_classes=self.dm.n_classes
        )

    def get_parameters(self, config):
        return self.glob_model.get_weights()

    def get_loglikelihood_ratio(self) -> float:
        eval_loss, _ = self.glob_model.evaluate(self.dm.X_val, self.dm.y_val)
        loss, _ = self.model.evaluate(self.dm.X_val, self.dm.y_val)
        
        return eval_loss - loss

    def fit(self, parameters, config):
        self.glob_model.set_weights(parameters)
        self.likelihood_ratio = self.get_loglikelihood_ratio()

        self.sel = config['sel']

        self.selected = str(self.cid) in config['selected_by_server'].split(',')

        if not self.last_participation is None:
            participate = self.last_participation

        parameters_to_response = parameters
        if config['rnd'] == 1:
            participate = True
        else:
            participate = bool(self.likelihood_ratio <= 0)
        if self.selected:
            if participate:
                if 'avg' in self.sel:
                    self.model.set_weights(parameters)
                    self.fit_avg(parameters, config)
                    parameters_to_response = self.model.get_weights()
                elif 'prox' in self.sel:
                    self.fit_prox(parameters, config)
                    parameters_to_response = self.model.get_weights()
                elif 'maxfl' in self.sel:
                    parameters_to_response = self.fit_maxfl(parameters, config)
            self.last_participation = participate

        self.local_model_size = np.sum([layer.nbytes for layer in self.model.get_weights()])
        return parameters_to_response, self.dm.X_train.shape[0], {'cid': self.cid, 'participate': participate, 'maxfl_w': self.maxfl_w}

    def fit_maxfl(self, parameters, config):
        epochs = 1
        batch_size = 32
        optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
        loss_fnc = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss = 0
        self.tmp_model.set_weights(copy.deepcopy(parameters))
        global_loss, global_accuracy = self.tmp_model.evaluate(self.dm.X_val, self.dm.y_val)
        for epoch in range(epochs):
            for i in range(0, len(self.dm.X_train), batch_size):
                with tf.GradientTape() as tape:
                    x_batch = self.dm.X_train[i:i + batch_size]
                    y_batch = self.dm.y_train[i:i + batch_size]
                    local_rep = self.tail(x_batch, training=True)
                    y_pred = self.head(local_rep, training=True)
                    local_loss = loss_fnc(y_batch, y_pred)
                grad = tape.gradient(local_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        self.maxfl_w = self.sigmoid(global_loss-self.q)
        delta_local = [
            self.model.get_weights()[i] - self.tmp_model.get_weights()[i] * self.maxfl_w
            for i in range(len(parameters))
        ]
        return delta_local

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit_avg(self, parameters, config):
        epochs = 1
        batch_size = 32
        optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
        loss_fnc = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss = 0
        for epoch in range(epochs):
            for i in range(0, len(self.dm.X_train), batch_size):
                with tf.GradientTape() as tape:
                    # Mini-batch
                    x_batch = self.dm.X_train[i:i + batch_size]
                    y_batch = self.dm.y_train[i:i + batch_size]

                    local_rep = self.tail(x_batch, training=True)
                    y_pred = self.head(local_rep, training=True)

                    local_loss = loss_fnc(y_batch, y_pred)

                grad = tape.gradient(local_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

    def fit_prox(self, parameters, config):
        """
            Baseado na implementação em pytorch do https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedProx.html
        """
        proximal_mu = 2.0
        epochs = 1
        optimizer = tf.keras.optimizers.Adam()
        batch_size = 32

        for epoch in range(epochs):
            for i in range(0, len(self.dm.X_train), batch_size):
                with tf.GradientTape() as tape:
                    x_batch = self.dm.X_train[i:i+batch_size]
                    y_batch = self.dm.y_train[i:i+batch_size]

                    y_pred = self.model(x_batch, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, y_pred)
                    loss = tf.reduce_mean(loss)

                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.model.trainable_variables, parameters):
                        proximal_term += tf.reduce_sum(tf.square(tf.norm(local_weights - global_weights, ord=2)))
                    
                    total_loss = loss + (proximal_mu / 2) * proximal_term

                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def evaluate(self, parameters, config):
        self.glob_model.set_weights(parameters)
        eval_loss, eval_accuracy = self.glob_model.evaluate(
            self.dm.X_test,
            self.dm.y_test,
        )
        val_loss, val_accuracy = self.glob_model.evaluate(
            self.dm.X_val,
            self.dm.y_val,
        )
        self.save_performance(eval_loss, eval_accuracy, val_loss, val_accuracy, config['rnd'])
        return eval_loss, self.dm.X_test.shape[0], {
            'eval_accuracy': eval_accuracy,
            'cid': self.cid,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'likelihood_ratio': self.likelihood_ratio
        }

    def save_performance(self, test_loss, test_accuracy, val_loss, val_accuracy, rnd):
        performance = {
            'loss': [test_loss],
            'accuracy': [test_accuracy],
            "val_loss": [val_loss],
            "val_accuracy": [val_accuracy], 
            'rnd': [rnd],
            'cid': [self.cid],
            'sel': [f"{self.sel}"],
            'selected': [self.selected],
            'participate': [self.last_participation],
            'likelihood': [self.likelihood_ratio],
            'size_data': [self.dm.X_train.shape[0]],
            'model_size': [self.local_model_size],
        }

        # Check if the file exists
        if os.path.exists(f'data/performance_cid_{self.cid}.csv'):
            df = pd.read_csv(f'data/performance_cid_{self.cid}.csv')
            df_to_save = pd.DataFrame(performance)
            df_to_save['size_data'] = self.dm.X_train.shape[0]
            df = pd.concat([df, pd.DataFrame(performance)], ignore_index=True)
        else:
            df = pd.DataFrame(performance)

        df.to_csv(f'data/performance_cid_{self.cid}.csv', index=False)
