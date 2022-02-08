import pyxel
import numpy as np

SQUARE = 16
class Rocket(object):
    def __init__(self):
        self.x = 5
        self.y = pyxel.height/2 - 8
        self.vel = 4
    
    def move(self):
        for key, u in zip([pyxel.KEY_RIGHT, pyxel.KEY_LEFT], [1, -1]):
            if pyxel.btnp(key, 1, 1):
                new_x = self.x + u*self.vel
                if new_x + SQUARE <= pyxel.width + 1 and new_x >= 0:
                    self.x = new_x

        for key, u in zip([pyxel.KEY_UP, pyxel.KEY_DOWN], [-1, 1]):
            if pyxel.btnp(key, 1, 1):
                new_y = self.y + u*self.vel
                if new_y + SQUARE <= pyxel.height + 1 and new_y >= 0:
                    self.y = new_y
                    
            
    def show(self):
        pyxel.blt(self.x, self.y, 0, 0, 0, SQUARE, SQUARE, pyxel.COLOR_BLACK)
        
        