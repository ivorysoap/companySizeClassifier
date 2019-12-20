import pandas as pd
import datetime
import pickle

def pickleFile(data, filename):
    """ Pickles the data into the specified filename. """
    pickle.dump(data, open(filename, 'wb'))
    return


def unpickleFile(filename):
    """ Take a pickle out of the jar. """
    return pickle.load(open(filename, 'rb'))


def cleanDataset(ds):
    """ Custom method for cleaning our dataset. """
    pd.to_numeric(ds['year_founded'], errors='coerce')  # Force year_founded column to numeric
    ds = ds.dropna()  # Drop all NaN rows
    return ds


def getPrettyTimestamp():
    """ Returns timestamp: 'YYYY-MM-DD_HHMM' """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(":", "").replace(" ", "_")
