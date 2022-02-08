import pyxel
import numpy as np
from itertools import product

SQUARE = 16
class Rocket(object):
    def __init__(self):
        self.x = 5
        self.y = 120/2 - 8
        self.vel = 4
    
    def move(self, action):
        for key, u in zip([3, 4], [1, -1]):
            if action == key:
                new_x = self.x + u*self.vel
                if new_x + SQUARE <= 120 + 1 and new_x >= 0:
                    self.x = new_x

        for key, u in zip([1, 2], [-1, 1]):
            if action==key:
                new_y = self.y + u*self.vel
                if new_y + SQUARE <= 120 + 1 and new_y >= 0:
                    self.y = new_y
                    
            
    def show(self):
        pyxel.blt(self.x, self.y, 0, 0, 0, SQUARE, SQUARE, pyxel.COLOR_BLACK)
        
    def state_mask(self):
        mask = np.zeros([120,120])
        for xs, ys in product(range(int(self.x),int(self.x+SQUARE)),range(int(self.y),int(self.y + SQUARE))):
            try:
                mask[xs,ys]=1
            except: 
                pass
        return mask