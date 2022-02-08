# First try on making a custom pyxel game "AI playable" (using OpenAI's gym.env)

For playing the game normally, you can use the comand:
```python
python Human_play\game.py
```

For training the AI you should run:
```python
python AI_play\gym_game.py
```

And for watching the AI play the game you can use
```python
python AI_play\AI_gameplay.py
```

Note, you should change the parameters in the `PPO` (model we are using for training the agent) for better results, consider using CUDA and training longer.

Also, make sure the name you are loading the model at line 18 of `AI_gameplay.py` is the same one you saved at the end of `gym_game.py`.

Please, contact me if you have any doubt.
