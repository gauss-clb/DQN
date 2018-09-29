from ale_python_interface import ALEInterface
import numpy as np
import cv2
import os
import shutil


def copyDir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)
    os.mkdir(src)


class AtariEmulator:
    
    def __init__(self, rom, visualization = False, save = False, windowName = 'AtariGame'):
        self.ale = ALEInterface()
        # self.ale.setInt(b'frame_skip', 1)
        self.ale.setInt(b"random_seed", 123)
        # self.ale.setFloat(b'repeat_action_probability', 0) # default = 0.25
        self.ale.loadROM(b'roms/' + rom)
        self.legalActions = self.ale.getMinimalActionSet()
        self.life_lost = False
        self.mode = 'train'
        self.visualization = visualization and not save
        self.windowName = windowName
        self.save = save
        self.totalReward = 0
        if self.visualization:
            cv2.namedWindow(self.windowName)
        elif self.save:
            self.index = 0
            self.bestReward = 0
            self.totalReward = 0
            if os.path.exists('result'):
                shutil.rmtree('result')
            if os.path.exists('best_result'):
                shutil.rmtree('best_result')
            if not os.path.exists('result'):
                os.mkdir('result')
            if not os.path.exists('best_result'):
                os.mkdir('best_result')


    def start(self):
        # In train mode: life_lost = True but game is not over, don't restart the game
        if self.mode == 'test' or not self.life_lost or self.ale.game_over():
            self.ale.reset_game()
        self.life_lost = False
        return cv2.resize(self.ale.getScreenGrayscale(), (84, 110))[26:]


    def isTerminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()


    def next(self, action): # index of action int legalActions
        lives = self.ale.lives() # the remaining lives
        reward = 0
        for i in range(4): # action repeat
            reward += self.ale.act(self.legalActions[action])
            self.life_lost = (lives != self.ale.lives())  # after action, judge life lost
            if self.mode == 'train' and self.life_lost:
                reward -= 1
            if self.isTerminal():
                break
        self.totalReward += reward
        state = self.ale.getScreenGrayscale()
        rgb_state = self.ale.getScreenRGB()
        if self.visualization:
            cv2.imshow(self.windowName, rgb_state)
            cv2.waitKey(10)
        elif self.save:
            cv2.imwrite(os.path.join('result', '%04d.png') % self.index, rgb_state)
            self.index += 1
            if self.isTerminal():
                print('Scores: %d, index: %d' % (self.totalReward, self.index))
                if self.totalReward > self.bestReward:
                    self.bestReward = self.totalReward
                    copyDir('result', 'best_result')
                self.index = 0
                self.totalReward = 0

        return cv2.resize(state, (84, 110))[26:], reward, self.isTerminal()


    def setMode(self, mode):
        self.mode = mode

    def randomStart(self, s_t):
        channels = s_t.shape[-1]
        self.start()
        for i in range(np.random.randint(channels, 30) + 1):
            s_t_plus_1, r_t, isTerminal = self.next(0)
            s_t[..., 0:channels-1] = s_t[..., 1:channels]
            s_t[..., -1] = s_t_plus_1
            if isTerminal:
                self.start()




