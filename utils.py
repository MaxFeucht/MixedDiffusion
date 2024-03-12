import os 

def create_dirs(**kwargs):

    # Check if directory for imgs exists
    for i in range(10000):
        imgpath = f'./imgs/{kwargs["dataset"]}_{kwargs["degradation"]}/run_{i}/'
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
            break
    
    modelpath = f'./models/{kwargs["dataset"]}_{kwargs["degradation"]}/'
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    return imgpath, modelpath