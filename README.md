
# 🐍 Snake AI — Deep Q-Learning

An AI-powered Snake game built with Python, Pygame, NumPy, and PyTorch.

The goal of this project is to train an artificial intelligence agent to play the classic Snake game using Deep Q-Learning (DQN). Instead of manually controlling the snake, the agent learns which action to take based on the current game state and the rewards it receives.
---
<img width="1919" height="1019" alt="Screenshot 2026-08-18 194417" src="https://github.com/user-attachments/assets/84e8b59d-ef93-4120-89fe-67ef4ac20848" />
---
## 🎮 Demo

The agent starts with little knowledge about the game and gradually learns through repeated gameplay.

During training, the AI:

1. Observes the current state of the snake.
2. Chooses an action.
3. Plays that action in the game.
4. Receives a reward or penalty.
5. Observes the new state.
6. Stores the experience in memory.
7. Uses previous experiences to improve the neural network.

Over time, the agent learns strategies that allow it to survive longer and collect more food.

## 🧠 How the AI Works

This project uses a neural network to approximate Q-values.

For every state, the network predicts the expected future reward for each possible action.

The three possible actions are:

```text
[1, 0, 0] → Move Straight
[0, 1, 0] → Turn Right
[0, 0, 1] → Turn Left
```

The neural network receives 11 values describing the current state of the snake.

```text
Input: 11 features
        ↓
Linear Layer
11 → 256
        ↓
ReLU
        ↓
Linear Layer
256 → 3
        ↓
Q-values for 3 actions
```

The action with the highest predicted Q-value is normally selected.

## 📊 State Representation

The agent does not see the entire game screen.

Instead, the game state is represented using 11 boolean features.

The state contains:

```text
1. Danger straight
2. Danger right
3. Danger left

4. Moving left
5. Moving right
6. Moving up
7. Moving down

8. Food is left
9. Food is right
10. Food is up
11. Food is down
```

For example, the state might look like:

```python
[
    0, 1, 0,
    0, 1, 0, 0,
    1, 0, 0, 1
]
```

This gives the AI enough information to understand:

* Where it is moving
* Whether it is about to collide
* Where the food is located

## 🧠 Neural Network

The model is implemented using PyTorch.

Architecture:

```text
Input Layer
11 neurons
    ↓
Hidden Layer
256 neurons
    ↓
ReLU Activation
    ↓
Output Layer
3 neurons
```

The three output neurons represent the Q-value of:

```text
Straight
Right
Left
```

The model does not directly output the action.

Instead, it outputs something similar to:

```text
[2.31, 0.72, 1.54]
```

The AI selects the action with the highest Q-value:

```text
Straight → 2.31
Right    → 0.72
Left     → 1.54

Chosen action → Straight
```

## 🎯 Q-Learning

The agent uses the Q-Learning update equation:

```text
Q_new = reward + γ × max(Q(next_state))
```

When the game is over:

```text
Q_new = reward
```

In this project:

```python
gamma = 0.9
```

The discount factor controls how much the AI values future rewards.

A higher gamma makes the agent more focused on long-term rewards.

## 🎁 Reward System

The game uses a simple reward system.

```text
Eat food       → +10
Die / collide  → -10
Normal move    →  0
```

This encourages the AI to:

* Find food
* Avoid collisions
* Stay alive
* Maximize its score

The reward is the main signal through which the agent learns.

## 🔍 Exploration vs Exploitation

The agent uses an epsilon-greedy strategy.

During early training, the AI performs more random actions.

```python
self.epsilon = 80 - self.n_games
```

If the random condition is satisfied, the AI chooses a random action.

Otherwise, it uses the neural network prediction.

Conceptually:

```text
Exploration
    ↓
Try random actions
    ↓
Discover useful strategies

Exploitation
    ↓
Use learned Q-values
    ↓
Choose the best known action
```

This balance is important.

If the AI only explores, it never properly uses what it has learned.

If it only exploits, it can get stuck with a bad strategy.

## 🧠 Experience Replay

The agent stores previous experiences in a replay memory:

```python
self.memory = deque(maxlen=100_000)
```

Each experience contains:

```text
(state, action, reward, next_state, done)
```

For example:

```text
Current State
      ↓
    Action
      ↓
   Reward
      ↓
 Next State
      ↓
   Game Over?
```

After every game, the agent samples experiences from memory and trains on them.

The batch size is:

```python
BATCH_SIZE = 1000
```

Experience replay helps reduce correlations between consecutive training samples and makes learning more stable.

## 🏗️ Project Structure

A clean version of the project can be organized like this:

```text
Snake-AI/
│
├── agent.py
├── model.py
├── snake_game.py
├── helper.py
├── model.pth
├── requirements.txt
└── README.md
```

### model.py

Contains:

```text
Linear_QNet
QTrainer
```

The neural network and training logic are implemented here.

### agent.py

Contains:

```text
Agent
```

The agent is responsible for:

* Creating the state
* Choosing actions
* Storing experiences
* Training short-term memory
* Training long-term memory

### snake_game.py

Contains the actual Snake game.

It handles:

* Snake movement
* Food generation
* Collision detection
* Score
* Game rendering
* Game reset

### helper.py

Contains the training visualization.

It plots:

```text
Score
Mean Score
```

while the AI is training.

## ⚙️ Hyperparameters

Current configuration:

```python
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
HIDDEN_SIZE = 256
INPUT_SIZE = 11
OUTPUT_SIZE = 3
```

These values control how the AI learns.

### Learning Rate

```python
LR = 0.001
```

Controls how quickly the neural network updates its weights.

### Discount Factor

```python
GAMMA = 0.9
```

Controls the importance of future rewards.

### Replay Memory

```python
MAX_MEMORY = 100_000
```

Controls how many previous experiences are stored.

### Batch Size

```python
BATCH_SIZE = 1000
```

Controls how many experiences are used during long-term training.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Snake-AI.git
cd Snake-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch pygame numpy matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Start training with:

```bash
python agent.py
```

The Snake game will open and the AI will start playing automatically.

You should see something similar to:

```text
Game 1 Score: 0 record: 0
Game 2 Score: 1 record: 1
Game 3 Score: 0 record: 1
Game 4 Score: 2 record: 2
...
```

As training progresses, the average score should generally improve.

## 💾 Saving the Model

Whenever the AI achieves a new record score, the model is saved:

```python
if score > record:
    record = score
    agent.model.save()
```

The model weights are stored in:

```text
model.pth
```

This allows the trained neural network to be reused later instead of training from scratch every time.

## 📈 Training Visualization

The project continuously plots:

```text
Score
Mean Score
```

The score shows the performance of individual games.

The mean score provides a better idea of the overall learning trend.

A typical training curve may look like:

```text
Score
  |
10|                 /\      /\
  |        /\      /  \ /\ /  \
  |   /\  /  \ /\ /    V
  |__/  \/    V  V
  +----------------------------> Games
```

The exact curve will vary because reinforcement learning is stochastic.

## 🔄 Training Loop

The complete training process can be summarized as:

```text
Start Game
    ↓
Get Current State
    ↓
Choose Action
    ↓
Perform Action
    ↓
Receive Reward
    ↓
Get New State
    ↓
Train Short-Term Memory
    ↓
Store Experience
    ↓
Game Over?
    │
    ├── No → Continue
    │
    └── Yes
          ↓
       Reset Game
          ↓
   Train Long-Term Memory
          ↓
    Save Best Model
          ↓
      Start Again
```

## 🧩 Short-Term Training

The agent trains immediately after every move:

```python
agent.train_short_memory(
    state_old,
    final_move,
    reward,
    state_new,
    done
)
```

This allows the agent to learn from its most recent experience.

## 🧠 Long-Term Training

When a game ends, the agent samples experiences from its memory:

```python
mini_sample = random.sample(
    self.memory,
    BATCH_SIZE
)
```

These experiences are then used to train the neural network.

This is called experience replay.

## 🛠️ Technologies Used

Python

PyTorch

Pygame

NumPy

Matplotlib

Deep Q-Learning

Reinforcement Learning

## 📚 Concepts Demonstrated

This project is useful for understanding several important AI and machine learning concepts:

```text
Reinforcement Learning
        ↓
Q-Learning
        ↓
Deep Q-Network
        ↓
Neural Networks
        ↓
Experience Replay
        ↓
Epsilon-Greedy Exploration
        ↓
Reward-Based Learning
```

It also demonstrates how an AI agent can learn behavior without being explicitly programmed with rules such as:

```text
"If food is left, go left."

"If wall is ahead, turn right."
```

Instead, the agent learns these behaviors through trial and error.


## 🎓 What I Learned From This Project

This project demonstrates the complete reinforcement-learning cycle:

```text
Observe
   ↓
Decide
   ↓
Act
   ↓
Receive Reward
   ↓
Learn
   ↓
Repeat
```

The most interesting part is that the Snake behavior is not hard-coded.

The AI starts with random behavior and gradually learns a strategy by interacting with its environment.



## 👨‍💻 Author

Built as a reinforcement-learning project to explore how neural networks can learn to play games through trial and error.

If you found this project useful or interesting, consider giving it a ⭐ on GitHub.

