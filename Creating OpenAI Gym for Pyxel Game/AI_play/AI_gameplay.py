import gym as gym

from rockets import Rocket
from meteoro import Meteoro
from utils import check_colision

import numpy as np
import pyxel

from stable_baselines3 import PPO

class GameEnv(gym.Env):
    def __init__(self):
        self.start_game()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                        shape=(120,120,1), dtype=np.uint8)
        self.model = PPO.load('jogador_decente.zip')
        self.state = self.reset()
        pyxel.init(120, 120)
        pyxel.load("jogo.pyxres")  
        pyxel.run(self.update, self.draw)
        
    def start_game(self):
        self.rocket = Rocket()
        self.GAME_OVER = False
        self.NEW_METEOR= 0
        self.meteoros = [Meteoro()]
        self.frame_game = 0
    
    def update(self):    
        action, _ = self.model.predict(self.state)
        self.state, _, _, _ = self.step(action)
    
    def state_mask(self):
        mask = np.zeros([120,120])
        mask += self.rocket.state_mask()
        for meteoro in self.meteoros:
                mask += meteoro.state_mask()
        mask = 255*((mask>0).astype(int))
        return np.array(mask).reshape(120,120,1)
    
    def score(self):
        return self.NEW_METEOR
            
    def step(self, action):
        if check_colision(self.rocket, self.meteoros):
            self.GAME_OVER=True
        
        if not self.GAME_OVER:        
            self.rocket.move(action)
            for meteoro in self.meteoros:
                meteoro.update()
            self.meteoros = [meteoro for meteoro in self.meteoros if meteoro.alive()]
            
            self.frame_game+=1
            if int(self.frame_game**(1.05))> self.NEW_METEOR*10:
                self.meteoros.append(Meteoro())
                self.NEW_METEOR+=1 
        
        state = self.state_mask()
        done = self.GAME_OVER
        info = {}
        if not self.GAME_OVER:
            reward = 1
        else:
            reward = -30
        
        return state, reward, done, info
        
    def reset(self):
        self.start_game()
        state = self.state_mask()
        return state
    
    def draw(self):
        pyxel.cls(pyxel.COLOR_BLACK)
        pyxel.bltm(0, 0, 0, 0, 0, pyxel.width, pyxel.height, pyxel.COLOR_BLACK)
        pyxel.text(pyxel.width-18, 5, str(self.score()), pyxel.COLOR_WHITE)
        
        for meteoro in self.meteoros:
            meteoro.show()
        self.rocket.show()
        
        if self.GAME_OVER:
            pyxel.text(pyxel.width/2-16, pyxel.height/2-4, 'GAME OVER', pyxel.COLOR_WHITE)
            
GameEnv()