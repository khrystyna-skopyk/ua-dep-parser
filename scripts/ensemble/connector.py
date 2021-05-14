

class Connector():
    def __init__(self, model, uas_weight, las_weight):
        self.model = model
        self.uas_weight = float(uas_weight) 
        self.las_weight = float(las_weight)