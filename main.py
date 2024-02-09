import numpy as np

LEARNING_RATE = 0.1
EPOCHS = 10

def unipolar_function(s, beta = 5.0):
    return 1 / (1 + np.exp(-s * beta))

def generate_weights(P, T):
    return np.random.rand(T.shape[0], P.shape[0])

def neural_network(P, T, weights, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    num_examples = P.shape[1]

    for epoch in range(epochs):
        for _ in range(num_examples):
            i = np.random.choice(num_examples)

            chosen_row = P[:, [i]]
            expected = T[:, [i]]

            y = unipolar_function(np.dot(weights, chosen_row))

            error = expected - y

            delta_weights = learning_rate * (np.dot(error, chosen_row.T))

            weights = weights + delta_weights

    return weights

def train():
    # Dane z instrukcji zadania
    P = np.array([
        [4.00, 2.00, -1.00],
        [0.10, -1.00, 3.50],
        [0.01, 2.00, 0.10],
        [-1.00, 2.50, -2.00],
        [-1.50, 2.00, 1.50]
    ])

    T = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    starting_weights = generate_weights(P, T)

    weights_after_training = neural_network(P, T, starting_weights)

    print(f"Wektor wejściowy P\n{P}\n")
    print(f"Wektor nauczyciela T\n{T}\n")
    print(f"Macierz wygenerowanych wag:\n{starting_weights.T}\n")
    print(f"Macierz wag po nauczeniu:\n{weights_after_training.T}\n")

    return weights_after_training

# Funkcja weryfikuje dzialanie metody train
def verify(weights_after_training):
    # Pierwszy zestaw weryfikujacy
    P = np.array([
        [2.0, -5.0, 4.0],
        [-1.5, 3.7, 0.2],
        [3.0, -0.01, 0.3],
        [2.5, -1.6, -1.4],
        [2.0, 1.1, -2.8]
    ])

    new_predictions = unipolar_function(np.dot(weights_after_training, P))
    print("Tabela klasyfikacji pierwszego zestawu weryfikującego")
    print("Pierwsza kolumna: ptak")
    print("Druga kolumna: ryba")
    print("Trzecia kolumna: ssak")
    print(np.round(new_predictions), "\n")

    # Drugi zestaw weryfikujacy
    P = np.array([
        [4.7, 2.0, -5.7],
        [0.26, -2.2, 3.7],
        [0.2, 4.0, -0.02],
        [-3, 3.7, -1.6],
        [-2.8, 2.6, 1.1]
    ])

    new_predictions = unipolar_function(np.dot(weights_after_training, P))
    print("Tabela klasyfikacji drugiego zestawu weryfikującego")
    print("Pierwsza kolumna: ssak")
    print("Druga kolumna: ptak")
    print("Trzecia kolumna: ryba")
    print(np.round(new_predictions))

    return 0

def main():
    print("---------------------------------------------------")
    print("------------------- Uczenie -----------------------")
    print("---------------------------------------------------")
    weights_after_training = train()

    print("---------------------------------------------------")
    print("------------------ Weryfikacja --------------------")
    print("---------------------------------------------------")
    verify(weights_after_training)


main()