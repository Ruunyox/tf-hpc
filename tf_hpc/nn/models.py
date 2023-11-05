import tensorflow as tf
from typing import Optional, List, Union, Callable, Tuple
from copy import deepcopy
from typing import Union, List, Dict


def CompileBuilder(
    optimizer: Optional[Union[str, tf.keras.optimizers.Optimizer]] = None,
    loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
    metrics: Optional[
        Union[str, tf.keras.metrics.Metric, List[Union[str, tf.keras.metrics.Metric]]]
    ] = None,
    loss_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    run_eagerly: Optional[bool] = None,
    steps_per_execution: Optional[Union[int, str]] = None,
    **kwargs,
) -> Dict:
    """Option wrapper for `tf.keras.Model.compile()`. See help(tf.keras.Model.compile)
    for more details.
    """
    out_dict = {
        "optimizer": optimizer,
        "loss": loss,
        "metrics": metrics,
        "loss_weights": loss_weights,
        "run_eagerly": run_eagerly,
        "steps_per_execution": steps_per_execution,
    }
    out_dict.update(kwargs)
    return out_dict


def FitBuilder(
    epochs: int = 1,
    verbose: Optional[Union[str, int]] = "auto",
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    class_weight: Optional[Dict[int, float]] = None,
    initial_epoch: int = 0,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    validation_freq: Union[int, List[int]] = 1,
    max_queue_size: int = 10,
    workers: int = 1,
    **kwargs,
) -> Dict:
    """Wrapper for `tf.keras.Model.fit()`. See `help(tf.keras.Model.fit)` for more details.
    Only supports tf.Dataset input-compatible options
    """
    out_dict = {
        "epochs": epochs,
        "verbose": verbose,
        "callbacks": callbacks,
        "class_weight": class_weight,
        "initial_epoch": initial_epoch,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "validation_freq": validation_freq,
        "max_queue_size": max_queue_size,
        "workers": workers,
    }
    out_dict.update(kwargs)
    return out_dict


class ModelWrapper(object):
    """Wrapper for `tf.keras.Model`"""

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)


class BasicImageModel(tf.keras.Model):
    """Template for basic image input model with more granular step methods. Expects tf.datasets.Dataset style input"""

    def __init__(self):
        super(BasicImageModel, self).__init__()

    def train_step(self, data: Dict) -> Dict:
        """Model training step"""
        if len(data) == 3:
            x, y, sample_weight = data["image"], data["label"], data["sample_weight"]
        else:
            x, y, sample_weight = data["image"], data["label"], None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data: Dict) -> Dict:
        """Model test/validation step"""
        if len(data) == 3:
            x, y, sample_weight = data["image"], data["label"], data["sample_weight"]
        else:
            x, y, sample_weight = data["image"], data["label"], None

        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def call(self, *args, **kwargs):
        raise NotImplementedError()


class FullyConnectedClassifier(BasicImageModel):
    """Fully connected, feed-forward image classifier

    Parameters
    ----------
    tag:
        `str` specifying model name
    out_dim:
        `int` specifying the number of classes
    activation:
        Valid tf.keras.activations `str` or `Callable`
        instance for the hidden layer activations
    class_activation
        Valid tf.keras.activations `str` or `Callable`
        instance for the class prediction activation.
    hidden_layers:
        `List[int]` of hidden layer dimensions for linear transforms
    """

    def __init__(
        self,
        tag: str,
        out_dim: int,
        activation: Union[str, Callable],
        class_activation: Union[str, Callable],
        hidden_layers: Optional[List[int]] = None,
    ):
        super(FullyConnectedClassifier, self).__init__()

        self.tag = tag
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        dense_names = [f"dense_{i}" for i in range(len(hidden_layers))]
        activation_names = [f"activation_{i}" for i in range(len(hidden_layers))]

        layers = []
        layers.append(
            tf.keras.layers.Dense(
                hidden_layers[0], activation=None, name=dense_names[0]
            )
        )
        layers.append(tf.keras.layers.Activation(activation, name=activation_names[0]))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(
                    tf.keras.layers.Dense(
                        hidden_layers[i], activation=None, name=dense_names[i]
                    )
                )
                layers.append(
                    tf.keras.layers.Activation(activation, name=activation_names[i])
                )
        layers.append(
            tf.keras.layers.Dense(out_dim, activation=None, name="class_dense")
        )
        layers.append(
            tf.keras.layers.Activation(class_activation, name="class_activation")
        )

        self.net = layers

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the model"""
        _, out_channels, pixel_x, pixel_y = x.shape
        x = tf.reshape(x, (-1, pixel_x * pixel_y * out_channels))
        for layer in self.net:
            x = layer(x)
        return x


class ConvolutionClassifier(BasicImageModel):
    """Simple Convolutional Neural Network with MaxPooling for classification

    Parameters
    ----------
    tag:
        `str` name for the model
    out_dim:
        `int` specifying the (non-batch) fully-connected network output dimension
    activation:
        Valid tf.keras.activations `str` or `Callable` instance for the hidden layer
        activations
    class_activation
        Valid tf.keras.activations `str` or `Callable` instance for the class
        prediction activation.
    hidden_layers:
        `List[int]` of hidden linear transform widths in the fully connected network
    conv_channels:
        `List[int]` specifying the series of channels for each convolution layer
    conv_kernels:
        `List[Union[List[int], List[Tuple[int]]]]` specifying the series of kernel sizes for each
        convolutional layer
    pooling_kernels:
        `List[Union[List[int], List[Tuple[int]]]]` specifying the series of kernel sizes for each
        pooling layer
    """

    def __init__(
        self,
        tag: str,
        out_dim: int,
        activation: Union[str, Callable],
        class_activation: Union[str, Callable],
        hidden_layers: Optional[List[int]] = None,
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[Union[List[int], List[Tuple[int]]]] = None,
        pooling_kernels: Optional[Union[List[int], List[Tuple[int]]]] = None,
    ):
        super(ConvolutionClassifier, self).__init__()

        self.tag = tag

        if conv_channels is None:
            conv_channels = [64, 64]
        if conv_kernels is None:
            conv_kernels = [2, 2]
        if pooling_kernels is None:
            pooling_kernels = [2, 2]
        if hidden_layers is None:
            hidden_layers = [64, 32]
        assert len(conv_channels) == len(conv_kernels) == len(pooling_kernels)

        conv_names = [f"conv_{i}" for i in range(len(conv_channels))]
        conv_pool_names = [f"conv_pool_{i}" for i in range(len(conv_channels))]
        conv_activation_names = [
            f"conv_activation_{i}" for i in range(len(conv_channels))
        ]

        conv_layers = []

        for i in range(0, len(conv_channels)):
            conv_layers.append(
                tf.keras.layers.Conv2D(
                    conv_channels[i],
                    conv_kernels[i],
                    name=conv_names[i],
                    activation=None,
                )
            )
            conv_layers.append(
                tf.keras.layers.MaxPool2D(pooling_kernels[i], name=conv_pool_names[i])
            )
            conv_layers.append(
                tf.keras.layers.Activation(activation, name=conv_activation_names[i])
            )

        self.convolutions = conv_layers

        dense_names = [f"dense_{i}" for i in range(len(hidden_layers))]
        activation_names = [f"activation_{i}" for i in range(len(hidden_layers))]

        layers = []
        layers.append(
            tf.keras.layers.Dense(
                hidden_layers[0], activation=None, name=dense_names[0]
            )
        )
        layers.append(tf.keras.layers.Activation(activation, name=activation_names[0]))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(
                    tf.keras.layers.Dense(
                        hidden_layers[i], activation=None, name=dense_names[i]
                    )
                )
                layers.append(
                    tf.keras.layers.Activation(activation, name=activation_names[i])
                )
        layers.append(
            tf.keras.layers.Dense(out_dim, activation=None, name="class_dense")
        )
        layers.append(
            tf.keras.layers.Activation(class_activation, name="class_activation")
        )

        self.net = layers

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the model"""
        x = tf.cast(x, tf.float32)
        for layer in self.convolutions:
            x = layer(x)

        _, out_channels, pixel_x, pixel_y = x.shape
        x = tf.reshape(x, (-1, pixel_x * pixel_y * out_channels))
        for layer in self.net:
            x = layer(x)
        return x
