# DQN

# Prerequisites

+ Python3.5
+ Numpy, Scipy, Matplotlib, Shutil
+ Tensorflow
+ [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)

# Training

```python
python main.py
```

**The 200 epoches will take more than 3 days, you can early stop it. The records of training process are in the `breakout.csv`.** 

# Testing

```python
python main.py --train=False
```

# Saving the best result 

```python
python main.py --train=False --save=True
```

**The best result is saved in the directory `best_result`, we can use ffmpeg to get a video.**

# Best result

![best_result](best.gif)


# References
* [simple_dqn](https://github.com/tambetm/simple_dqn)

# License
MIT License