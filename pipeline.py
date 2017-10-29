from vectorize import vectorize

def main():

    DATA_PATH = '' # Path of directory containing 'aps' files
    LABEL_PATH = '' # Path of csv file with labels

    body_zone_vectors = vectorize(DATA_PATH, LABEL_PATH)

if __name__ == '__main__' : main()
