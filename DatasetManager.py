import flwr_datasets as fld
import tensorflow as tf
import numpy as np
import typing as T
import datasets as D
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class DatasetManager:
    def __init__(self, dataset: str, n_partitions: int, non_iid: bool, cid: int):
        self.cid = cid
        self.dataset_name = dataset
        train_p, val_p, test_p = self.get_partitioners(n_partitions, non_iid)
        self.fd: fld.FederatedDataset = fld.FederatedDataset(dataset=dataset, partitioners = {'train': train_p, 'test': test_p})
        self.fd_val: fld.FederatedDataset = fld.FederatedDataset(dataset=dataset, partitioners = {'train': val_p})

        self.cd_applied = False
        try:
            fig, _, _ = fld.visualization.plot_comparison_label_distribution(
                partitioner_list=[self.fd.partitioners["train"], self.fd_val.partitioners["train"], self.fd.partitioners["test"]],
                label_name="label",
                subtitle=f"Comparison of Partitioning Schemes",
                titles=["Train distribution", "Validation distribution", "Test distribution"],
                legend=True,
                verbose_labels=True,
            )
            fig.savefig('partition_distributions.png', format='png', dpi=300, bbox_inches='tight')
        except:
            print("Error while plotting partition distributions")
        self._X_train, self._y_train, self._X_val, self._y_val, self._X_test, self._y_test = self.get_partition(self.cid)
    
    @property
    def X_train(self):
        if self.cd_applied:
            return self.X_train_incoming
        return self._X_train
    
    @property
    def y_train(self):
        if self.cd_applied:
            return self.y_train_incoming
        return self._y_train
    
    @property
    def X_test(self):
        if self.cd_applied:
            return self.X_test_incoming
        return self._X_test
    
    @property
    def y_test(self):
        if self.cd_applied:
            return self.y_test_incoming
        return self._y_test
    
    @property
    def X_val(self):
        return self._X_val
    
    @property
    def y_val(self):
        return self._y_val
    
    def get_full_data(self) -> D.Dataset:
        train = self.fd.load_split('train').with_format('tf')
        try:
            test = self.fd.load_split('test').with_format('tf')
            
            self.d = D.concatenate_datasets([train, test])
        except:
            self.d = train
        X, y = self._features_and_labels(self.d)
        return X, y

    def get_full_test(self):
        test = self.fd.load_split('test').with_format('tf')
        X, y = self._features_and_labels(test)
        return X, y

    def get_partitioners(self, n_partitions: int, non_iid: bool):
        if non_iid:
            return (
                fld.partitioner.DirichletPartitioner(
                    num_partitions = n_partitions,
                    alpha = 0.1, # LEMBRAR DE MUDAR PARA OUTROS DATASETS
                    partition_by = 'label',
                    seed = 0
                ),
                fld.partitioner.DirichletPartitioner(
                    num_partitions = n_partitions,
                    alpha = 0.1, # LEMBRAR DE MUDAR PARA OUTROS DATASETS
                    partition_by = 'label',
                    seed = 42
                ),
                fld.partitioner.IidPartitioner(
                    num_partitions = n_partitions
                )
            )
        return (
            fld.partitioner.IidPartitioner(
                num_partitions = n_partitions
            ),
            fld.partitioner.IidPartitioner(
                num_partitions = n_partitions
            ),
            fld.partitioner.IidPartitioner(
                num_partitions = n_partitions
            )
        )
    
    def split_train_validation(self, dataset: D.Dataset):
        train, val = fld.utils.divide_dataset(dataset, division=[0.8, 0.2])
        return train.with_format("tf"), val.with_format("tf")
    
    def get_partition(self, cid: int)-> T.List[D.Dataset]:
        train_partition = self.fd.load_partition(cid, 'train').with_format("tf")
        try:
            self.test_partition = self.fd.load_partition(cid, 'test').with_format("tf")
        except:
            train_partition, self.test_partition = self.split_train_validation(train_partition)    
        self.train_partition = train_partition
        self.val_partition = self.fd_val.load_partition(cid, 'train').with_format("tf")

        keys = list(train_partition.features.keys())
        self.n_classes = len(train_partition.features[keys[-1]].names)
        train_images = self.train_partition[keys[0]]
        train_labels = self.train_partition[keys[-1]]
        val_images = self.val_partition[keys[0]]
        val_labels = self.val_partition[keys[-1]]
        test_images = self.test_partition[keys[0]]
        test_labels = self.test_partition[keys[-1]]

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def _features_and_labels(self, data: D.Dataset):
        data = data.with_format('tf')
        keys = list(data.features.keys())
        train_images = data[keys[0]]
        train_labels = data[keys[-1]]
        if self.dataset_name == 'flwrlabs/office-home':
            train_images = tf.convert_to_tensor(np.array([cv2.resize(np.array(img), (128, 128)) for img in train_images]))
        else:
            train_images = tf.convert_to_tensor(np.array([np.array(img) for img in train_images]))
        return train_images, train_labels