import numpy as np
import matplotlib.pyplot as plt


def scores_logistic_regression(model,
                               X_train: np.array,
                               Y_train: np.array,
                               X_test: np.array
                               ) -> np.array:

    # Fit model
    model.fit(X_train, Y_train)

    # Make predictions on the test set
    scores = model.predict(X_test)

    return scores


def scores_random_forest(model,
                         X_train: np.array,
                         Y_train: np.array,
                         X_test: np.array
                         ) -> np.array:

    # Fit model
    model.fit(X_train, Y_train)

    # Make predictions on the test set
    scores = model.predict(X_test)

    return scores


def scores_neural_network(model,
                          X_train: np.array,
                          Y_train: np.array,
                          X_test: np.array,
                          epochs=300,
                          batch_size=128,
                          verbose=0,
                          visualize_training=True
                          ) -> np.array:

    # Fit model
    model_history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Make predictions on the test set
    scores = (model.predict(X_test) > 0.5).astype("int32")

    # Plot model history
    if visualize_training:
        plt.figure(figsize=(5, 4))

        # Plot data
        plt.plot(model_history.history['loss'], color='C0', linestyle='-', linewidth=1.5, alpha=1, label='Training')

        # Options
        plt.title(r'Training history', fontsize=15)
        plt.xlabel(r'Epochs', fontsize=15, labelpad=8)
        plt.ylabel(r'Loss', fontsize=15, labelpad=8)
        plt.legend(loc='upper right', fontsize=12)
        plt.tick_params(which='both', direction='out', bottom=True, left=True, right=True, labelsize=14)
        plt.show()

    return scores
