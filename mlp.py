# Imports
import numpy as np
from reader import Reader
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from metr import purity, rand_index
from mlp_init import mlp_set_labels, get_new_seed_points


def mlp_main(SAMPLES):
    feature_vector_length = 784
    num_classes = 10

    # Load the data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train[:SAMPLES]

    X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
    X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert target classes to categorical ones
    Y_train_1d = Y_train

    # Set the input shape
    input_shape = (feature_vector_length,)
    print(f'Feature shape: {input_shape}')

    # Create the model
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(num_classes, activation='softmax'))

    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Start fitting
    # train_temp = np.random.randint(10, size=Y_train.shape[0]).astype('float32')
    # numpy.savetxt('random_labels.csv', train_temp, delimiter=',')
    # train_temp = numpy.loadtxt('random_labels.csv', delimiter=',')
    train_temp = mlp_set_labels(X_train, num_classes)
    Y_train_temp = to_categorical(train_temp[:SAMPLES], num_classes)

    iterations = 0

    while True:
        model.fit(X_train, Y_train_temp, epochs=5, batch_size=64, verbose=1, validation_split=0.2)

        # Predict
        prediction = model.predict_classes(X_train, verbose=0)

        # Update Labels by Distance From Centers
        new_centers = get_new_seed_points(X_train, prediction)
        improved_labels = mlp_set_labels(X_train, num_classes, new_centers)

        prediction_Y = to_categorical(improved_labels, num_classes)

        iterations += 1

        if (Y_train_temp == prediction_Y).all():
            break

        Y_train_temp = prediction_Y

    # data_reader = Reader('data')
    # images_count_to_show = 25
    # for i in range(num_classes):
    #     cluster_images = [X_train[index] for index in np.where(prediction == i)[0][:images_count_to_show]]
    #     data_reader.plot_images(cluster_images[:images_count_to_show], images_count_to_show)

    return prediction, Y_train_1d

#
# print("SOM Scores:")
# pprint(som_scores)

# dist = cdist(Y_train_temp, prediction_Y, metric="euclidean")

# Test the model after training
# test_results = model.evaluate(X_test, Y_test, verbose=1)
# print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')


# temp_train_labels = np.zeros_like(Y_train, dtype='float32')
# def randomize_init_labels(target):
#     def random_label(row):
#         row[np.random.randint(10)] = 1.0
#         return row
#
#     return np.array(list(map(random_label, target)))
#
#
# temp_train_labels = randomize_init_labels(temp_train_labels)
