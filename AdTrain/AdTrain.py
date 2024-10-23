"""
An implementation of the Adversarial training method from the following work:

Y. Wei and R. Zheng, "Handling Device Heterogeneity in Wi-Fi based Indoor Positioning Systems," 
IEEE Conference on Computer Communications Workshops (INFOCOM), 2020

Approach adapted to classification

Sub-project refers to the implementation of AdLoc

Notes to contributors:
- Keep all dependencies within sub-project
- Usable functions should be importable from outside sub-project
- Enable high flexibility when possible from outside sub-project
"""

# TODO:
# Function(s) to build model as per paper
# Ability to provide custom model structure as input

from tensorflow import keras
from typing import List


def build_ad_train(
    layers: List,
    input_noise: float,
    label_noise: float,
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=1e-4),
    loss: keras.losses.Loss = keras.losses.sparse_categorical_crossentropy,
    name: str = "AdTrain",
    metrics=["accuracy"],
) -> keras.Sequential:
    """Creates a compiled keras model

    Parameters
    ----------
    layers : List
        List of MLP layers that will be the body of the model.
        First item in the list should be a keras input layer.
        Should not have any Gaussian Noise layers
    label_noise : float, optional
        noise inserted into the location labels
    label_noise : float, optional
        noise inserted into the location labels
    """

    # TODO: Update docStr

    # check if first layer is input
    assert "input" in layers[0].name, "First layer should be input"

    model = keras.Sequential(
        layers=[
            layers[0],  # original input layer
            keras.layers.GaussianNoise(input_noise, name="InputNoise"),
            *layers[1:],  # add layers provided
            keras.layers.GaussianNoise(label_noise),  # noise after output layer
        ],
        name=name,
    )

    # Compile model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    return model
