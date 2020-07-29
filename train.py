import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_policy_net(action_space_size, state_space_size):
    model = Sequential()
    model.add(Dense(256, input_dim=state_space_size,
                    kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(action_space_size, activation='softmax'))
    return model


def create_value_net(state_space_size):
    model = Sequential()
    model.add(Dense(256, input_dim=state_space_size,
                    kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


def sample_action(action_space_size, probs, use_max=False):
    if use_max:
        return np.argmax(probs)
    else:
        return np.random.choice(action_space_size, p=probs/probs.sum())


def eval(model, env, max_eps, action_space_size):
    total_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            action_dist = model(tf.convert_to_tensor(
                [state], dtype=tf.float32))
            action = sample_action(
                action_space_size, action_dist.numpy()[0], use_max=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / max_eps
    return avg_reward


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append([discounted_reward])
    return discounted_rewards[::-1]


def train(max_eps=1000, gamma=0.99):
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]
    policy_net = create_policy_net(action_space_size, state_space_size)
    old_policy_net = create_policy_net(action_space_size, state_space_size)
    value_net = create_value_net(state_space_size)
    policy_optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    value_optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    for eps in range(max_eps):
        state = env.reset()
        done = False
        rewards, states, actions = [], [], []
        while not done:
            prob = policy_net(tf.convert_to_tensor(
                [state], dtype=tf.float32)).numpy()[0]
            action = sample_action(action_space_size, prob)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
        old_probs = old_policy_net(
            tf.convert_to_tensor(states, dtype=tf.float32))
        with tf.GradientTape(persistent=True) as tape:
            probs = policy_net(tf.convert_to_tensor(states, dtype=tf.float32))
            vals = value_net(tf.convert_to_tensor(states, dtype=tf.float32))
            log_old_probs = tf.math.log(
                tf.clip_by_value(old_probs, 1e-10, 1.0-1e-10))
            q_vals = tf.convert_to_tensor(
                compute_discounted_rewards(rewards, gamma), dtype=tf.float32)
            advantage = q_vals - vals
            value_loss = advantage ** 2
            log_probs = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0-1e-10))
            ratio = -tf.math.exp(log_probs - log_old_probs)
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)
            action_onehot = tf.one_hot(
                actions, action_space_size, dtype=tf.float32)
            min_ratio = tf.minimum(ratio, clipped_ratio)
            policy_loss = -(log_probs * action_onehot) * advantage
            entropy_loss = -tf.reduce_sum(probs * log_probs)
            loss = tf.reduce_mean(0.5 * value_loss) + \
                tf.reduce_mean(policy_loss) + 0.01 * entropy_loss
        old_policy_net.set_weights(policy_net.get_weights())
        policy_grads = tape.gradient(loss, policy_net.trainable_weights)
        val_grads = tape.gradient(loss, value_net.trainable_weights)
        policy_optimizer.apply_gradients(
            zip(policy_grads, policy_net.trainable_weights))
        value_optimizer.apply_gradients(
            zip(val_grads, value_net.trainable_weights))
        del tape
        eval_score = eval(policy_net, eval_env, 10, action_space_size)
        print(
            'Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()
    print('Done!')


if __name__ == '__main__':
    train()
