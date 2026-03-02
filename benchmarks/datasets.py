import pickle
import os

class Dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = os.path.abspath(path)
        self.tasks = None

    def get_tasks(self):
        with open(self.path, "rb") as file:
            self.tasks = pickle.load(file)
            return self.tasks

current_dir = os.path.dirname(os.path.abspath(__file__))

VirtualHome = Dataset("virtualhome317", os.path.join(current_dir, "virtualhome", "virtualhome317.pkl"))