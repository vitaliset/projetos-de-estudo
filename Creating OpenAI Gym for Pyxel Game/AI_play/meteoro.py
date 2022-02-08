import pyxel
import numpy as np
from itertools import product

RADIUS = 2

class Meteoro(object):
    def __init__(self):
        self.x = 120 + 5*RADIUS
        self.y = np.random.randint(10, 120-10)
        self.vel_x = -np.random.randint(3,5)
    
    def show(self):
        pyxel.circ(self.x, self.y, RADIUS, 15)
        
    def update(self):
        self.x += self.vel_x

    def alive(self):
        return (self.x >= 0)

    def state_mask(self):
        mask = np.zeros([120,120])
        for xs, ys in product(range(self.x-RADIUS,self.x+RADIUS),range(self.y-RADIUS,self.y + RADIUS)):
            try:
                mask[xs,ys]=1
            except: 
                pass
        return mask