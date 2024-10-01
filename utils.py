import pickle

def pload(path):
    with open(path,'rb') as file:
        return pickle.load(file)

def psave(object,path):
    with open(path,'wb') as file:
        pickle.dump(object,file)
