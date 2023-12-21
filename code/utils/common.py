import os
import pickle


def load_pickle(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump_pickle(path, data, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, **kwargs)


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory {path} already exists.")
    except OSError:
        print(f"Creation of the directory {path} failed.")
    else:
        print(f"Successfully created the directory {path}.")

    return path

