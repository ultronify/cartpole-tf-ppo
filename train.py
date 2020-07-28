import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def create_policy_net(action_space_size, state_space_size):
    model = Sequential()
    model.add(Dense(128, input_dim=state_space_size,
                    kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(action_space_size, activation='softmax'))
    return model


def create_value_net(state_space_size):
    model = Sequential()
    model.add(Dense(128, input_dim=state_space_size,
                    kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


def sample_action(prob, action_space_size, use_max=False):
    if use_max:
        return np.argmax(prob)
    else:
        return np.random.choice(action_space_size, p=prob/prob.sum())


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append([discounted_reward])
    return discounted_rewards[::-1]


def eval(model, env, max_eps, action_space_size):
    total_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            action_dist = model(tf.convert_to_tensor([state]))
            action = sample_action(
                action_space_size, action_dist.numpy()[0], use_max=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / max_eps
    return avg_reward


def train(max_eps=1000, gamma=0.99):
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]
    policy_net = create_policy_net(action_space_size, state_space_size)
    value_net = create_value_net(state_space_size)
    optimizer = Adam(learning_rate=1e-3)
    rewards, states, old_probs = [], [], []
    for eps in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            prob = policy_net(tf.convert_to_tensor(
                [state], dtype=tf.float32)).numpy()[0]
            action = sample_action(prob, action_space_size)
            next_state, reward, done, _ = env.step(action)
            old_probs.append(prob)
            rewards.append(reward)
            states.append(state)
            state = next_state
        with tf.GradientTape(persistent=True) as tape:
            probs = policy_net(tf.convert_to_tensor(states, dtype=tf.float32))
            vals = value_net(tf.convert_to_tensor(states, dtype=tf.float32))
            ratio = tf.math.exp(tf.convert_to_tensor(old_probs, dtype=tf.float32) - probs)
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)
            q_vals = tf.convert_to_tensor(
                compute_discounted_rewards(rewards, gamma), dtype=tf.float32)
            advantage = q_vals - vals
            loss = tf.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_grads = tape.gradient(loss, policy_net.trainable_weights)
        optimizer.apply_gradients(
            zip(policy_grads, policy_net.trainable_weights))
        val_grads = tape.gradient(loss, value_net.trainable_weights)
        optimizer.apply_gradients(zip(val_grads, value_net.trainable_weights))
        del tape
        eval_score = eval(policy_net, eval_env, 10, action_space_size)
        print(
            'Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()
    print('Done!')


if __name__ == '__main__':
    train()
