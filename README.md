Flappy Bird Deep Q-network
===============

Train a neural network to play Flappy Bird using Deep Q-learning.

Flappy Bird made in python from [here](https://github.com/sourabhv/FlapPyBird) 

Setup 
---------------------------

1. Install Python 3.x (recommended) from [here](https://www.python.org/download/releases/) (Or use your preffered package manager)

2. _Optional_: Setup a virtual environment from [here](https://pypi.org/project/virtualenv/)

3. Clone the repository: 
    ```bash
   $ git clone https://github.com/yenchenlin/DeepLearningFlappyBird.git
   ```
4. Install dependencies:

   ```bash
   $ pip install -r requirements.txt
   ```

5. To run training network (q-learning):

   ```bash
   $ python deep_q_train.py
   ```
   To start training from the beginning, delete data/training_values_resume.json and files in model_weights.
   (Make sure to back up because it takes a long time to train.)

Description
---------------------------
Inputs 4 images of the game into a deep q-network based off https://github.com/nikitasrivatsan/DeepLearningVideoGames.

References
---------------------------
1. Flappy Bird made in python from [here](https://github.com/sourabhv/FlapPyBird) 

2. Explanations and deep q-model from [here](https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial)

3. Code used:
   1. [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
   2. [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch)
   3. [pytorch-flappy-bird](https://github.com/hardlyrichie/pytorch-flappy-bird)

4. Q-learning models from above based from [here](https://github.com/nikitasrivatsan/DeepLearningVideoGames)
