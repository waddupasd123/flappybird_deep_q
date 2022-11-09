import torch
import torch.nn as nn
import numpy as np
import cv2
import flappy_mod as Flappy
from conv_model import FlappyConv
from random import random, randint, sample
import matplotlib.pyplot as plt

INITIAL_EPSILON = 0.1       # starting value of epsilon
FINAL_EPSILON = 0.0001      # final value of epsilon
NUM_ITERS = 2000000
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.000006

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train():
    model = FlappyConv()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    Flappy.launch()
    movementInfo = Flappy.initialSetup()
    gameInfo = Flappy.mainGameSetup(movementInfo)
    
    image, gameInfo, death = Flappy.mainGame(gameInfo)
    image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    image = image[None, :, :].astype(np.float32)
    



    image = torch.from_numpy(image)
    model.to(device)
    image = image.to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :]

    replay_memory = []
    iter = 0
    score = 0
    losses = []
    iterations = []
    while iter < NUM_ITERS:
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = FINAL_EPSILON + ((NUM_ITERS - iter) * (INITIAL_EPSILON - FINAL_EPSILON) / NUM_ITERS)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction)

        quit = Flappy.action(gameInfo, action)
        next_image, gameInfo, death = Flappy.mainGame(gameInfo)
        if death:
            reward = -1
            Flappy.launch()
            movementInfo = Flappy.initialSetup()
            gameInfo = Flappy.mainGameSetup(movementInfo)
        elif gameInfo['score'] > score:
            reward = 1
            score = gameInfo['score']
        else: 
            reward = 0.1

        next_image = cv2.cvtColor(cv2.resize(next_image, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, next_image = cv2.threshold(next_image,1,255,cv2.THRESH_BINARY)
        next_image = next_image[None, :, :].astype(np.float32)
        next_image = torch.from_numpy(next_image)
        next_image = next_image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        replay_memory.append([state, action, reward, next_state, death])
        if len(replay_memory) > REPLAY_MEMORY_SIZE:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), BATCH_SIZE))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            reward_batch = reward_batch.to(device)
            next_state_batch = next_state_batch.to(device)
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + GAMMA * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        print(loss)
        losses.append(loss.item())
        iterations.append(iter)

        optimizer.step()


        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            NUM_ITERS,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        #writer.add_scalar('Train/Loss', loss, iter)
        plt.figure(1)
        plt.title("Train/Loss")
        plt.ylabel("Train")
        plt.ylabel("Loss")
        plt.plot(iterations, losses)
        plt.show(block=False)
        plt.pause(0.001)
        #writer.add_scalar('Train/Epsilon', epsilon, iter)
        #writer.add_scalar('Train/Reward', reward, iter)
        #writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        if (iter+1) % 1000000 == 0:
            torch.save(model, "data/data" + str(iter + 1))
    torch.save(model, "data/data")


if __name__ == "__main__":
    train()