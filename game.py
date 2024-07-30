from config import *
import random
import torch
import time
from copy import deepcopy
from torch.autograd import Variable
import torch.nn as nn
from queue import Queue
import torch.nn.functional as F


def genFigures(cnt):
    choice = [x for x in range(len(FIGURES))]
    random.shuffle(choice)
    return sorted(choice[:cnt])


class Game:
    def __init__(self):
        self.score = None
        self.state = None
        self.reset()

    def copy(self):
        g = Game()
        g.state = deepcopy(self.state)
        g.score = self.score
        return g

    def canPlace(self, cx, cy, figureNumber):
        figure = FIGURES[figureNumber]
        if len(figure) + cx > HEIGHT or len(figure[0]) + cy > WIDTH:
            return False
        for x in range(len(figure)):
            for y in range(len(figure[0])):
                if self.state[cx + x][cy + y] and figure[x][y]:
                    return False
        return True

    def checkState(self):
        cntHorizontal, cntVertical = [], []
        for i in range(HEIGHT):
            f = True
            for j in range(WIDTH):
                if self.state[i][j] == 0:
                    f = False
            if f:
                cntHorizontal.append(i)
        for j in range(WIDTH):
            f = True
            for i in range(HEIGHT):
                if self.state[i][j] == 0:
                    f = False
            if f:
                cntVertical.append(j)
        for i in cntHorizontal:
            for j in range(WIDTH):
                self.state[i][j] = 0
        for j in cntVertical:
            for i in range(HEIGHT):
                self.state[i][j] = 0
        return len(cntHorizontal) * HEIGHT + len(cntVertical) * WIDTH + len(cntHorizontal) * len(
            cntVertical) * HEIGHT * WIDTH / 4

    def setFigure(self, figureNumber, cx, cy):
        figure = FIGURES[figureNumber]
        if len(figure) + cx > HEIGHT or len(figure[0]) + cy > WIDTH:
            return False
        for x in range(len(figure)):
            for y in range(len(figure[0])):
                if self.state[cx + x][cy + y] and figure[x][y]:
                    return False
        for x in range(len(figure)):
            for y in range(len(figure[0])):
                self.state[cx + x][cy + y] |= figure[x][y]
        nw = self.checkState()
        self.score += nw
        return True

    def printState(self):
        print('-' * 5, 'State of the game', '-' * 5)
        print(self.score)
        for i in range(HEIGHT):
            s = ''
            for j in range(WIDTH):
                s += str(self.state[i][j])
            print(s)

    def convertToTensor(self):
        result = []
        for i in range(HEIGHT):
            for j in range(WIDTH):
                result.append(self.state[i][j] - 0.5)
        return Variable(torch.tensor(result), requires_grad=True)

    def reset(self):
        self.state = [[0] * WIDTH for _ in range(HEIGHT)]
        self.score = 0

    def generateNextFigure(self):
        return genFigures(1)[0]


def makeStep(g, figureNumber, cx, cy):
    t = g.copy()
    was = t.score

    if not t.setFigure(figureNumber, cx, cy):
        return False, 0, t

    reward = t.score - was

    return True, reward, t
