import keras as K
import tensorflow as tf

SEED = 42
class ModelManager:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_model(self, input_shape, n_classes):
        if self.model_type == 'cnn':
            return self.get_cnn(input_shape, n_classes)
        elif self.model_type == 'dnn':
            return self.get_mlp(input_shape, n_classes)
        else:
            raise ValueError('Invalid model type')
    
    def get_cnn(self, input_shape, n_classes):
        tail_layers = [
            K.layers.Conv2D(
                filters = 256,
                kernel_size = (3, 3),
                activation = 'relu',
                input_shape = input_shape,
                kernel_initializer=K.initializers.RandomNormal(stddev=0.01),
            ),
            K.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),
            K.layers.BatchNormalization(),
            K.layers.Dropout(0.25),
            K.layers.Conv2D(
                filters = 128,
                kernel_size = (1, 1),
                activation = 'relu',
                kernel_initializer=K.initializers.RandomNormal(stddev=0.01),
            ),
            K.layers.BatchNormalization(),
            K.layers.MaxPooling2D(
                pool_size=(1, 1)
            ),
            K.layers.Dropout(0.25),
            K.layers.Conv2D(
                filters = 128,
                kernel_size = (3, 3),
                activation = 'relu',
                kernel_initializer=K.initializers.RandomNormal(stddev=0.01),
            ),
            K.layers.BatchNormalization(),
            K.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),
            K.layers.Dropout(0.25),
            K.layers.Conv2D(
                filters = 32,
                kernel_size = (2, 2),
                activation = 'relu',
                kernel_initializer=K.initializers.RandomNormal(stddev=0.01),
            ),
            K.layers.BatchNormalization(),
            K.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),
            K.layers.Dropout(0.25),
            
        ]
        
        head_layers = [
            K.layers.Dropout(0.25),
            K.layers.Flatten(),
            K.layers.Dense(
                units = n_classes,
                activation = 'softmax',
                kernel_initializer=K.initializers.RandomNormal(stddev=0.01),
            )
        ]
        
        input = K.layers.Input(shape = input_shape)
        tail = self.create_block(input, tail_layers, 'tail')
        head = self.create_block(tail, head_layers, 'head')

        
        model = K.models.Model(inputs=input, outputs=head, name='cnn')
        model_tail = K.models.Model(inputs=input, outputs=tail, name='tail_output')
        model_head = K.models.Model(inputs=tail, outputs=head, name='head_output')
        
        model.compile(
            optimizer = K.optimizers.Adam(),
            loss = K.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        return model_tail, model_head, model
    
    def create_block(self, input, layers, block_name):
        x = input
        for layer in layers:
            layer._name = f"{block_name}" # Ta assim pra evitar nomes com inteiros
            x = layer(x)
        return x

    def get_mlp(self, input_shape, n_classes):
        tail_layers = [
            K.layers.Flatten(),
            K.layers.Dense(
                256,
                activation='relu',
                kernel_initializer=K.initializers.LecunNormal(seed=SEED),
            ),
            tf.keras.layers.Dropout(0.25),
            K.layers.Dense(
                128,
                activation='relu',
                kernel_initializer=K.initializers.LecunNormal(seed=SEED),
            ),
        ]
        head_layers = [
            K.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=K.initializers.LecunNormal(seed=SEED),
            ),
            tf.keras.layers.Dropout(0.25),
            K.layers.Dense(
                32,
                activation='relu',
                kernel_initializer=K.initializers.LecunNormal(seed=SEED),
            ),
            K.layers.Dense(
                n_classes,
                activation='softmax',
                kernel_initializer=K.initializers.LecunNormal(seed=SEED),
            )
        ]
        
        input = K.layers.Input(shape = input_shape)
        tail = self.create_block(input, tail_layers, 'tail')
        head = self.create_block(tail, head_layers, 'head')

        
        model = K.models.Model(inputs=input, outputs=head, name='dnn')
        model_tail = K.models.Model(inputs=input, outputs=tail, name='tail_output')
        model_head = K.models.Model(inputs=tail, outputs=head, name='head_output')

        model.compile(
            optimizer = K.optimizers.SGD(clipvalue=1.0),
            loss = K.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        return model_tail, model_head, model
