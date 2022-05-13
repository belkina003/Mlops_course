import src

RAW_DATA_PATH = "data/raw/train.csv"
NORMAL_DATA_PATH = "data/interim/after_normilize.csv"
DUMMY_DATA_PATH = "data/interim/with_dummy.csv"

if __name__ == '__main__':
    print('11')
    src.normilize(RAW_DATA_PATH, NORMAL_DATA_PATH)
    print('22')
    src.add_features(NORMAL_DATA_PATH, DUMMY_DATA_PATH)
