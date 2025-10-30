import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, sigmoid, relu

# Softmax test for neural network utils
def test_my_softmax(target):
    z = np.array([1., 2., 3., 4.])
    a = target(z)
    atf = tf.nn.softmax(z)
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"

    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"

    print("\033[92mAll tests passed (softmax).")


# NN model architecture test
def test_model(target, classes, input_size):
    target.build(input_shape=(None, input_size))

    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.layers[0].input.shape == (None, input_size), \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"

    expected = [[Dense, (None, 25), relu],
                [Dense, (None, 15), relu],
                [Dense, (None, classes), linear]]

    for i, layer in enumerate(target.layers):
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"

    print("\033[92mAll tests passed (model structure).")


# Linear regression cost function test
def compute_cost_test(target):
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2
    initial_b = 3.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 0, f"Case 1: Cost must be 0 for a perfect prediction but got {cost}"

    # Case 2
    initial_w = 2.0
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 2, f"Case 2: Cost must be 2 but got {cost}"

    # Case 3
    x = np.array([1.5, 2.5, 3.5, 4.5, 1.5]).T
    y = np.array([4, 7, 10, 13, 5]).T
    initial_w = 1
    initial_b = 0.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 15.325), f"Case 3: Cost must be 15.325 but got {cost}"

    # Case 4
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 10.725), f"Case 4: Cost must be 10.725 but got {cost}"

    # Case 5
    y = y - 2
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 4.525), f"Case 5: Cost must be 4.525 but got {cost}"

    print("\033[92mAll tests passed (cost function).")


# Linear regression gradient test
def compute_gradient_test(target):
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4.5, 8.5, 12.5, 16.5]).T
    initial_w = 2.
    initial_b = 0.5
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    assert dj_db == 0.0, f"Case 1: dj_db is wrong: {dj_db} != 0.0"
    assert np.allclose(dj_dw, 0), f"Case 1: dj_dw is wrong: {dj_dw} != 0.0"

    # Case 2 
    y = np.array([6, 9, 12, 15]).T
    initial_w = 1.5
    initial_b = 1
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    assert dj_db == -2, f"Case 2: dj_db is wrong: {dj_db} != -2"
    assert np.allclose(dj_dw, -10.0), f"Case 2: dj_dw is wrong: {dj_dw} != -10.0"   

    print("\033[92mAll tests passed (gradient function).")
