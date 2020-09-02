import matplotlib.pyplot as plt

class Plot:

    def __init__(self, data_points, color1, color2, marker1, marker2):
        self.data_points = data_points
        self.color1 = color1
        self.color2 = color2
        self.marker1 = marker1
        self.marker2 = marker2

    def __call__(self):
        ###
        