from sklearn_som.som import SOM


def som_main(data):
    train_data = data[:, :784]

    som = SOM(m=30, n=30, dim=784)

    som.fit(train_data)

    predicted_labels = som.predict(train_data)

    return predicted_labels
