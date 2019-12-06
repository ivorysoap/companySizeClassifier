from tools import pickleFile
from tools import unpickleFile

def main():

    model = getModel()

    model.predict("Try")



def getModel(filename = "model/Model_2019-12-04_15:44"):
    """ Unpickle and load machine learning model. """
    return unpickleFile(filename)


if __name__ == "__main__":
    main()
