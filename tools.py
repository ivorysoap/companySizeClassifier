import pickle

def pickleFile(data, filename):
    """ Pickles the data into the specified filename. """
    pickle.dump(data, open(filename, 'wb'))
    return


def unpickleFile(filename):
    """ Take a pickle out of the jar. """
    return pickle.load(open(filename, 'rb'))
