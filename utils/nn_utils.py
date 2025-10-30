import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_nn_model(input_dim=400, output_dim=10):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    return model

def compile_and_train(model, X, y, epochs=40, learning_rate=0.001):
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return history

def predict_with_softmax(model, X):
    logits = model.predict(X)
    return tf.nn.softmax(logits).numpy()
