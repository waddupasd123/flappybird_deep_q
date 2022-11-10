import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import flappy_mod as Flappy
from conv_model import FlappyConv
from random import random, randint, sample
import matplotlib.pyplot as plt

# Adjustable values
INITIAL_EPSILON = 0.1       # starting value of epsilon
FINAL_EPSILON = 0.0001      # final value of epsilon
NUM_ITERS = 2000000
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.000006

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

def train():
    # Model setup
    model = FlappyConv()
    if os.path.exists("model_weights/flappy.pth"):
        model.load_state_dict(torch.load("model_weights/flappy.pth"))
        model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    # Game setup
    Flappy.launch()
    movementInfo = Flappy.initialSetup()
    gameInfo = Flappy.mainGameSetup(movementInfo)
    
    # Convert screenshot of game to numpy array
    image, gameInfo, death = Flappy.mainGame(gameInfo)
    image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    image = image[None, :, :].astype(np.float32)
    
    # Input image into deep q network
    image = torch.from_numpy(image)
    model.to(device)
    image = image.to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :]

    replay_memory = []
    losses = []
    iterations = []
    iter = 0
    score = 0
    episode_len = 0
    while iter < NUM_ITERS:
        # 2 Output values
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

        # Perform action
        quit = not Flappy.action(gameInfo, action)
        if (quit):
            break

        # Next Fram
        image, gameInfo, death = Flappy.mainGame(gameInfo)

        # Rewards
        if death:
            reward = -1
            # Relaunch
            Flappy.launch()
            movementInfo = Flappy.initialSetup()
            gameInfo = Flappy.mainGameSetup(movementInfo)
        elif gameInfo['score'] > score:
            reward = 1
            score = gameInfo['score']
        else: 
            reward = 0.1

        # Next state
        image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
        image = image[None, :, :].astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], image))[None, :, :, :]

        # Save important values
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
        loss = loss_func(q_value, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        iterations.append(iter)

        state = next_state

        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            NUM_ITERS,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))

        if iter % 25000 == 0:
            torch.save(model, "model_weights/flappy_" + str(iter) + ".pth")

        iter += 1

        episode_len += 1
        if (death):
            print("Start Episode", len(model.episodes) + 1)
            model.episodes.append(episode_len)
            plot_durations(model.episodes)
            episode_len = 0
        
    plt.savefig('training_results.png')
    torch.save(model, "model_weights/flappy.pth")


def plot_durations(episodes):
    """Plot durations of episodes and average over last 100 episodes"""
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episodes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)



if __name__ == "__main__":
    train()