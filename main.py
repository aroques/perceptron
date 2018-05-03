from random import choice
from numpy import dot, random
from dataset import get_training_data_from_txt_file


def main():
    num_columns, num_rows, training_data = get_training_data_from_txt_file()

    w = random.rand(3)
    errors = []
    eta = 0.2
    iterations = 100

    for i in range(iterations):
        x, expected = choice(training_data)
        result = dot(w, x)
        error = expected - sign(result)
        errors.append(error)
        w += eta * error * x

    for x, _ in training_data:
        result = dot(x, w)
        print("{}: {} -> {}".format(x[:num_columns], result, sign(result)))


def sign(x):
    return -1 if x < 0 else 1


if __name__ == '__main__':
    main()
