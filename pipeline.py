from vectorize import vectorize

def main():

    DATA_PATH = '' # Path of directory containing 'aps' files
    LABEL_PATH = '' # Path of csv file with labels

    xs, ys = vectorize(DATA_PATH, LABEL_PATH)

if __name__ == '__main__' : main()
