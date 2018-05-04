from random import choice
from numpy import dot, random
from dataset import get_training_data_from_txt_file, line_to_int_list


def main():
    num_columns, num_rows, training_data = get_training_data_from_txt_file()

    iterations = 100
    w = random.rand(3)
    eta = 0.2

    no_error_count = 0

    print('Training the perceptron...')

    for i in range(iterations):
        x, expected = choice(training_data)
        result = dot(w, x)
        error = expected - sign(result)

        if error == 0:
            no_error_count += 1
        else:  # Reset the error count
            no_error_count = 0

        w += eta * error * x

        error_rate = compute_error_rate(training_data, w)
        print('Iteration {:2}: Error Rate = {:2}%'.format(i, error_rate * 100))

        if no_error_count == 30:
            print('No error was recorded 30 consecutive times, so the perceptron is trained.')
            print('30 is an arbitrarily chosen number and may need to be adjusted.')
            break

    print('\nTesting each sample in the training data...')
    for x, _ in training_data:
        result = dot(x, w)
        print("Sample {}: Result: {} -> Class: {}".format(x[:num_columns], result, sign(result)))

    print('\nEntering a loop to query the perceptron. Press ctrl-c at anytime to exit.')

    while True:
        sample = input('Enter a sample ({} numbers separated by a space): '.format(num_columns))
        try:
            sample = line_to_int_list(sample)
            sample.append(1)  # Append bias
        except ValueError:
            print('Input was not {} numbers separated by a space. Please try again. '.format(num_columns))
            continue
        result = dot(sample, w)
        print("Sample {}: Result: {} -> Class: {}".format(sample[:num_columns], result, sign(result)))


def compute_error_rate(training_data, w):
    num_misclassified = count_num_misclassified(training_data, w)
    return num_misclassified / float(len(training_data))


def count_num_misclassified(training_data, w):
    num_incorrect = 0
    for x, expected in training_data:
        result = dot(x, w)
        error = expected - sign(result)
        if error != 0:
            num_incorrect += 1
    return num_incorrect


def sign(x):
    return -1 if x < 0 else 1


if __name__ == '__main__':
    main()
