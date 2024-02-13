import argparse

# global variables
MIN = 0.45
MAX = 0.9


def limited_ratio_type(arg):
    '''
    Type function for argparse - a float within MIN and MAX
    '''

    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN or f > MAX:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX) + " and > " + str(MIN))
    return f


def limited_variance_type(arg):
    '''
    type function for argparse - a float within MIN and MAX
    '''

    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN or f > MAX:
        return 0
    return f


def parse_args():
    '''
    builds a parser for 
    command line arguments

    :returns: args values
    '''

    parser = argparse.ArgumentParser()

    # pipeline
    parser.add_argument("--reduction", "-rd", default=0, type=limited_variance_type, help="choose pca variance between {MIN} and {MAX} for feature extraction, or skip feature extraction")
    parser.add_argument("--classification", "-cl", default='bayes', choices=['None','bayes','discriminative','knn'], help="choose classification model")
    parser.add_argument("--allknn", "-ak", action="store_true", help="if knn is the classification model it computes for all possible k values and returns the best")
    parser.add_argument("--features", "-ft", default='41', type=str, help="choose data file based on number of features")
    parser.add_argument("--ratio", "-rt", default=0.7, type=limited_ratio_type, help=f"a ratio between {MIN} and {MAX} for deciding how large the training set will be")
    parser.add_argument("--seed", "-sd", default=0, type=int, help="choose the seed for the random dataset split generator")
    
    # others
    parser.add_argument("--save", "-sv", action="store_true", help="save information about execution in an output file")


    return parser.parse_args()