from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_ann(projected_data, labels, num_classes):
    X = projected_data.T
    y = to_categorical(labels, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

    test_acc = model.evaluate(X_test, y_test)[1]
    return model, test_acc
