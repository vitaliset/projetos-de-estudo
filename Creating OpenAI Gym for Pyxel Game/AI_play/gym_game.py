import gym as gym

from rockets import Rocket
from meteoro import Meteoro
from utils import check_colision

import numpy as np

class GameEnv(gym.Env):
    def __init__(self):
        # self.start_game()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                        shape=(120,120,1), dtype=np.uint8)
    
    def start_game(self):
        self.rocket = Rocket()
        self.GAME_OVER = False
        self.NEW_METEOR= 0
        self.meteoros = [Meteoro()]
        self.frame_game = 0
    
    def update(self):    
        # self.step(action)
        pass
    
    def state_mask(self):
        mask = np.zeros([120,120])
        mask += self.rocket.state_mask()
        for meteoro in self.meteoros:
                mask += meteoro.state_mask()
        mask = 255*((mask>0).astype(int))
        # return np.array([mask]*3).T.reshape(120,120,3)#np.array(3*[mask]).reshape(3,120,120)
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
    
    
env = GameEnv()

import matplotlib.pyplot as plt
# plt.imshow(env.reset())
# plt.show()

# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# env = DummyVecEnv([lambda: env])
# env = VecFrameStack(env, 4, channels_order='last')

# state = env.reset()
# for i in range(10):
#     state, reward, done, info = env.step([4])

# plt.figure(figsize=(20,16))
# for idx in range(state.shape[3]):
#     plt.subplot(1,4,idx+1)
#     plt.imshow(state[0][:,:,idx])
# plt.show()

from stable_baselines3 import PPO
# from nilvo import TrainAndLoggingCallback

model = PPO('CnnPolicy', env, verbose=1, 
            learning_rate=0.0001, 
            n_steps=512) 

model.learn(total_timesteps=100000)
                            # 1000000
model.save('teste')

# NVidia GeForce 920M