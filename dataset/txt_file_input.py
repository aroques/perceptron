import argparse
from numpy import array


def get_training_data_from_txt_file() -> tuple:
    """Returns training data from text file.

    :return: See parse_txt_file.
    """
    args = get_cmd_ln_arguments()
    return parse_txt_file(args.filename)


def parse_txt_file(file):
    """
    Parses a user supplied text file for data.
    Args:
        file: The name of the text file that contains data.
    Returns:
        num_columns: The number of columns in the training data set
        num_rows: The number of rows in the training data set
        training_data: Training data (a list of integer lists)
    """
    training_data = []
    with open(file) as f:
        first_line = f.readline()
        num_columns = int(first_line[0])
        num_rows = int(first_line[2])
        for line in f:
            data_with_classification = get_data_with_classification(line)
            training_data.append(data_with_classification)

    return num_columns, num_rows, training_data


def get_data_with_classification(line):
    """Get a (numpy array, integer classification) tuple

    :param line: A string of integers. Ex: '1 3 5\n'
    :return: (numpy array, integer classification) tuple
    """
    data = line_to_int_list(line)
    classification = data.pop()
    bias_weight = 1
    data.append(bias_weight)
    return array(data), classification


def line_to_int_list(line):
    """
    Args:
        line: A string of integers. Ex: '1 3 5\n'
    Returns:
        A list of integers. Ex: [1, 3, 5]
    """
    data = line.split(' ')
    data = filter(None, data)
    data = [int(x.strip('\n')) for x in data]
    return data


def get_cmd_ln_arguments():
    """
    Returns:
        args: An object that contains command line argument data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help='name of file that contains data')
    args = parser.parse_args()
    return args
