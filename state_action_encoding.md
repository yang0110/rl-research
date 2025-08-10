Encoders in reinforcement learning (RL) are used to represent a state-action pair in a way that is useful for a policy or value function. The goal is to transform the raw data into a compact and meaningful feature vector. The specific encoding method depends heavily on the nature of the state and action spaces.

State-Action Pair Encoding

The most common approach is to concatenate the encoded state and action vectors and then feed this combined vector into a neural network.

1. State Encoding

The state can be encoded in various ways:

    Continuous State: For a state represented by a vector of continuous values (e.g., position, velocity), the vector can be fed directly into a neural network.

    Discrete State: A one-hot encoding is often used, where a vector with a length equal to the number of possible states has a '1' at the index corresponding to the current state and '0's elsewhere.

    Image State: Convolutional Neural Networks (CNNs) are employed to process image data and extract features into a fixed-size vector. This is common in video games or robotic vision tasks.

    Licensed by Google

2. Action Encoding

The action is encoded based on whether it is discrete or continuous:

    Continuous Action: Similar to a continuous state, the action vector can be used directly as input.

    Discrete Action: One-hot encoding is typically used, creating a vector where only the element corresponding to the chosen action is '1'.

3. Combining State and Action

After encoding, the state and action vectors, let's call them Senc​ and Aenc​, are concatenated to form a single input vector [Senc​,Aenc​]. This combined vector is then passed to a neural network, often a Multi-Layer Perceptron (MLP), which learns to map this representation to a value (e.g., the Q-value) or another useful output.

Examples of Encoding

1. Discrete State and Discrete Action

Consider a grid world where the state is the agent's position (e.g., row and column) and the action is the direction of movement (e.g., up, down, left, right).

    State Encoding: If the grid is 5x5, there are 25 possible states. A state like position (2, 3) can be one-hot encoded into a vector of length 25.

    Action Encoding: The four actions (up, down, left, right) can be one-hot encoded into a vector of length 4.

    Combined Vector: The concatenated vector would have a length of 25+4=29.

2. Continuous State and Continuous Action

Imagine a robot arm trying to reach a target. The state could be the joint angles and velocities (e.g., a vector of 10 values), and the action could be the torques applied to the joints (e.g., a vector of 5 values).

    State and Action Encoding: Both are already in a numerical vector format, so they can be used directly. No specific encoding is needed beyond normalization.

    Combined Vector: The state vector (length 10) and action vector (length 5) are simply concatenated to form a vector of length 10+5=15. This vector is then fed into a neural network.

3. Image State and Discrete Action

In a classic game like Atari's Pong, the state is the game screen (an image), and the action is moving the paddle up or down.

    State Encoding: A CNN would process the image frame (e.g., 84x84 pixels) to produce a fixed-size feature vector (e.g., length 256).

    Action Encoding: The two actions (up, down) are one-hot encoded into a vector of length 2.

    Combined Vector: The feature vector from the CNN and the action vector are concatenated to form a vector of length 256+2=258. This combined vector is then passed to a fully connected neural network to predict the Q-value for that state-action pair.