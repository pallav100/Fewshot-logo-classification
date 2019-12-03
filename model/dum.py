import os
def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    print(classes)

get_current_classes('/home/pallav_soni/pro/model/dum.txt')
