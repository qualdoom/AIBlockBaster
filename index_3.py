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
        self.state = [[0] * WIDTH for _ in range(HEIGHT)]
        self.score = 0

    def copy(self):
        g = Game()
        g.state = deepcopy(self.state)
        g.score = self.score
        return g

    def checkState(self):
        cnt1, cnt2 = [], []
        for i in range(HEIGHT):
            f = True
            for j in range(WIDTH):
                if self.state[i][j] == 0:
                    f = False
            if f:
                cnt1.append(i)
        for j in range(WIDTH):
            f = True
            for i in range(HEIGHT):
                if self.state[i][j] == 0:
                    f = False
            if f:
                cnt2.append(j)
        for i in cnt1:
            for j in range(WIDTH):
                self.state[i][j] = 0
        for j in cnt2:
            for i in range(HEIGHT):
                self.state[i][j] = 0
        return len(cnt1) * HEIGHT + len(cnt2) * WIDTH + len(cnt1) * len(cnt2) * HEIGHT * WIDTH // 4

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


def cnt_comps(state):
    cnt = 0
    used = [[0] * WIDTH for _ in range(HEIGHT)]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if used[i][j] or not state[i][j]:
                continue
            cnt += 1
            used[i][j] = 1
            q = Queue()
            q.put([i, j])
            while not q.empty():
                i1, j1 = q.get()
                for k in range(len(DIRECTIONS)):
                    x, y = i1 + DIRECTIONS[k][0], j1 + DIRECTIONS[k][1]
                    if 0 <= x < WIDTH and 0 <= y < HEIGHT and not used[x][y] and state[x][y]:
                        q.put([x, y])
                        used[x][y] = 1
    return cnt


class Neuro(nn.Module):
    def __init__(self, ):
        super(Neuro, self).__init__()
        self.fc1 = nn.Linear(WIDTH * HEIGHT, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def checkScore(self, g):
        result = []
        for i in range(HEIGHT):
            for j in range(WIDTH):
                result.append(g[i][j] - 0.5)
        '''for i in range(HEIGHT):
            for j in range(WIDTH):
                cnt = 0
                for k in range(len(DIRECTIONS)):
                    x, y = i + DIRECTIONS[k][0], j + DIRECTIONS[k][1]
                    if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
                        cnt += g[x][y]
                result[cnt] += 1
        result[5] = cnt_comps(g)'''
        return Variable(torch.tensor(result), requires_grad=True)

    def forward(self, x):
        x = self.checkScore(x.state)
        x = F.relu(self.fc1(x))
        # print(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x)
        return x

    def checkState(self, t, x, y, figureId):
        g = t.copy()
        was = g.score
        if not g.setFigure(figureId, x, y):
            return False, 0, g
        return True, g.score - was, g

gamma = 0.99

net = Neuro()

print(net)
criterion = nn.HuberLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
best = 0
data = []
DATA_LEN = 10
FIGURES_COUNT = 1000
for i in range(DATA_LEN):
    data.append([])
    for j in range(FIGURES_COUNT):
        data[i].append(genFigures(1)[0])
for epoch in range(10000):
    print('Run', epoch)
    g = Game()
    was = 0
    while True:
        figureID = genFigures(1)[0]
        dif, mx = 0, 0
        i1, j1 = -1, -1
        Fnd = False
        for i in range(HEIGHT):
            for j in range(WIDTH):
                good, reward, t = net.checkState(g, i, j, figureID)
                if not good:
                    continue
                sc = net(t) * gamma + reward
                if not Fnd or sc > mx:
                    mx, dif, i1, j1 = sc, reward, i, j
                Fnd = True
        if not Fnd:
            #print(mx, dif, i1, j1)
            print(g.score)
            # if epoch % 100 == 0:
                # time.sleep(0)
            break
        optimizer.zero_grad()
        # print(net(g), mx * 0.99 + dif)
        loss = criterion(net(g), mx.detach())
        # print(loss)
        loss.backward()
        optimizer.step()
        g.setFigure(figureID, i1, j1)
        if g.score != was:
            was = g.score
        best = max(was, best)
    print('Best', best)
