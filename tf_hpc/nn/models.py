import tensorflow as tf
from typing import Optional, List, Union, Callable
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
    """Wrapper for YAML configs. Only supports tf.Dataset input-compatible options"""
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
    """Wrapper for YAML configs"""

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)


class FullyConnectedClassifier(tf.keras.Model):
    def __init__(
        self,
        tag: str,
        in_dim: int,
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

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data['image'], data['label'], data['sample_weight']
        else:
            x, y, sample_weight = data['image'], data['label'], None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data['image'], data['label'], data['sample_weight']
        else:
            x, y, sample_weight = data['image'], data['label'], None

        y_pred = self(x, training=True)  # Forward pass
        # (the loss function is configured in `compile()`)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def call(self, x, training=False):
        """Forward pass through the model"""
        _, out_channels, pixel_x, pixel_y = x.shape
        x = tf.reshape(x, (-1, pixel_x * pixel_y * out_channels))
        for layer in self.net:
            x = layer(x)
        return x
