import os 

def create_dir(**kwargs):

    # Check if directory for imgs exists
    for i in range(10000):
        path = f'./imgs/{kwargs["dataset"]}_{kwargs["degradation"]}/run_{i}/'
        if not os.path.exists(path):
            os.makedirs(path)
            break

    return path