import numpy as np

def task000():

    # Creare array
    array1 = np.array([1, 2, 3])
    print("Array1:", array1)

    array2 = np.zeros((2, 3))
    print("Array2 (filled with zeros):")
    print(array2)

    array3 = np.ones((3, 2))
    print("Array3 (filled with ones):")
    print(array3)

    array4 = np.empty((2, 2))
    print("Array4 (empty array):")
    print(array4)

    # Manipolare array
    print("\nManipulating arrays:")
    array5 = np.array([[1, 2], [3, 4]])
    print("Original array:")
    print(array5)

    print("Transposed array:")
    print(np.transpose(array5))

    # Operazioni matematiche
    print("\nMathematical operations:")
    array6 = np.array([5, 6, 7])
    print("Array6:", array6)

    print("Sum of array6:", np.sum(array6))
    print("Mean of array6:", np.mean(array6))
    print("Element-wise square root of array6:", np.sqrt(array6))

    # Indicizzazione e selezione
    print("\nIndexing and selection:")
    array7 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original array:")
    print(array7)

    print("Selecting second row:", array7[1])
    print("Selecting element at position (1, 2):", array7[1, 2])
    print("Selecting elements from the first two rows:", array7[:2])

    # Algebra lineare
    print("\nLinear algebra:")
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])

    print("Matrix1:")
    print(matrix1)
    print("Matrix2:")
    print(matrix2)

    print("Matrix multiplication:")
    print(np.dot(matrix1, matrix2))

    # Trasformate
    print("\nTransforms:")
    array8 = np.array([1, 2, 3, 4])
    print("Original array:")
    print(array8)

    print("Discrete Fourier Transform:")
    print(np.fft.fft(array8))

    # Generazione di numeri casuali
    print("\nRandom number generation:")
    print("Random numbers from a uniform distribution:")
    print(np.random.rand(3, 3))

    print("Random numbers from a normal distribution:")
    print(np.random.randn(2, 2))

def task001():
    # TODO to implement
    print("do nothing")


def plot_histogram(prices):
    # TODO to implement
    print("do nothing")

def task002():
    # TODO to implement
    print("do nothing")


def task003():
    # TODO to implement
    print("do nothing")

def task004():
    # TODO to implement
    print("do nothing")

def task005():
    # TODO to implement
    print("do nothing")

if __name__ == "__main__":
    # Basic Operation
    task000()

    # Matrix Operations: performs product between two matrices
    # task001()

    # Load a dataset of housing prices, calculate the average price, and plot a histogram of the prices
    # task002()

    # FFT filtering signal
    # task003()

    # SQUEEZE function (introduction to torch
    # task004()

    # Data exchange between Numpy and other library
    # task005()

