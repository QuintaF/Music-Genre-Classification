import argparse

# global variables
MIN = 0.5
MAX = 0.9


def range_limited_float_type(arg):
    '''
    Type function for argparse - a float within .5 and .9
    '''

    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN or f > MAX:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX) + "and > " + str(MIN))
    return f


def parse_args():
    '''
    builds a parser for 
    command line arguments

    :returns: args values
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--reduction", "-rd", default='None', choices=['None','pca'], help="choose feature extraction method, or skip feature extraction")
    parser.add_argument("--classification", "-cl", default='bayes', choices=['None','bayes','discriminative','knn'], help="choose classification model")
    parser.add_argument("--allknn", "-ak", action="store_true", help="if knn is the classification model it computes for all possible k values and returns the best")
    parser.add_argument("--features", "-ft", default='41', type=str, help="choose data file based on number of features")
    parser.add_argument("--ratio", "-rt", default=0.7, type=range_limited_float_type, help=f"a ratio between {MIN} and {MAX} for deciding how large the training set will be with respect to the test set")


    return parser.parse_args()