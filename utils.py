def load_data(path : str):
    with open(path) as file:
        return file.read()

def save_data(path : str, data : str):
    with open(path, "w") as f:
        f.write(data)
