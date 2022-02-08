import pyxel
import numpy as np

RADIUS = 2

class Meteoro(object):
    def __init__(self):
        self.x = pyxel.width + 2*RADIUS
        self.y = np.random.randint(10, pyxel.height-10)
        self.vel_x = -np.random.randint(3,5)
    
    def show(self):
        pyxel.circ(self.x, self.y, RADIUS, 15)
        
    def update(self):
        self.x += self.vel_x

    def alive(self):
        return (self.x >= 0)
