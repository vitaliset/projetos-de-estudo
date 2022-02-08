
import pyxel
# import numpy as np
# from itertools import product

from rockets import Rocket
from meteoro import Meteoro
from utils import check_colision

class Game():
    
    def __init__(self):
        pyxel.init(120, 120)
        pyxel.load("jogo.pyxres")  
        self.start_game()
        pyxel.run(self.update, self.draw)

    def start_game(self):
        self.rocket = Rocket()
        self.GAME_OVER = False
        self.NEW_METEOR= 0
        self.meteoros = [Meteoro()]
        self.frame_game = 0
    
    def update(self):    
        if check_colision(self.rocket, self.meteoros):
            self.GAME_OVER=True
        
        if not self.GAME_OVER:        
            self.rocket.move()
            for meteoro in self.meteoros:
                meteoro.update()
            self.meteoros = [meteoro for meteoro in self.meteoros if meteoro.alive()]
            
            self.frame_game+=1
            if int(self.frame_game**(1.05))> self.NEW_METEOR*10:
                self.meteoros.append(Meteoro())
                self.NEW_METEOR+=1

    def draw(self):
        pyxel.cls(pyxel.COLOR_BLACK)
        pyxel.bltm(0, 0, 0, 0, 0, pyxel.width, pyxel.height, pyxel.COLOR_BLACK)
        pyxel.text(pyxel.width-18, 5, str(self.score()), pyxel.COLOR_WHITE)
        
        for meteoro in self.meteoros:
            meteoro.show()
        self.rocket.show()
        
        if self.GAME_OVER:
            pyxel.text(pyxel.width/2-16, pyxel.height/2-4, 'GAME OVER', pyxel.COLOR_WHITE)

    def score(self):
        return self.NEW_METEOR
    
Game()