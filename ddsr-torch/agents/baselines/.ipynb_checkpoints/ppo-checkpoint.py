import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils import layer_init 
from tqdm import tqdm
from encoder_models import * 
from rl_models import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)  
    parser.add_argument('--num_steps', type=int, default=128)  
    parser.add_argument('--num_envs', type=int, default=16)  
    parser.add_argument('--encoder_name', type=int, default='mlp')  
    parser.add_argument('--rep_dim', type=int, default=256)  
    parser.add_argument('--hidden_dim', type=int, default=256)  
    parser.add_argument('--res_block_num', type=int, default=1)  
    parser.add_argument('--batch_size', type=int, default=0)  
    parser.add_argument('--minibatch_size', type=int, default=512)  
    parser.add_argument('--episode_num', type=int, default=2000) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--epoch_num', type=float, default=10) 
    parser.add_argument('--gamma', type=float, default=0.999) 
    parser.add_argument('--gae_lambda', type=float, default=0.95) 
    parser.add_argument('--clip_coef', type=float, default=0.1) 
    parser.add_argument('--clip_vloss', type=bool, default=True) 
    parser.add_argument('--ent_coef', type=float, default=0.01) 
    parser.add_argument('--vf_coef', type=float, default=0.5) 
    parser.add_argument('--max_grad_norm', type=float, default=0.5) 
    parser.add_argument('--target_kl', type=float, default=None) 
    return parser.parse_args()

class PPOAgent(nn.Module):
    def __init__(self, encoder, critic, actor):
        super().__init__()
        
        self.encoder = encoder
        self.critic = critic 
        self.actor = actor
        
    def get_rep(self, x):
        rep = self.encoder(x)
        return rep

    def get_value(self, x):
        rep = self.encoder(x)
        value = self.critic(rep)
        return value
    
    def get_action_and_value(self, x, action=None):
        rep = self.encoder(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(x)
        entropy = probs.entropy()
        log_prob = probs.log_prob(action)
        return action, log_prob, entropy, value

    
if __name__ == "__main__":
    args = parse_args()
    
    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print('------ device', device)
    
    # create envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs_space = envs.single_obs_space
    action_space = envs.single_action_space
    
    print('------ obs_space', obs_space)
    print('------ action_space', action_space)
    print('------ action_num', action_space.n)
    obs_dim = obs_space.shape()[0]
    print('------ obs_dim', obs_dim)
    # create agent modules 
    encoder_name = args.encoder_name
    hidden_dim = args.hidden_dim
    rep_dim = args.rep_dim 
    res_block_num = args.res_block_num

    if encoder_name == 'mlp':
        encoder = MLPEncoder(obs_dim, hidden_dim, rep_dim)
    if encoder_name == 'pixel':
        encoder = PixelEncoder()
    if encoder_name = 'resnet':
        encoder = ResNetEncoder()
        
    critic = ValueNet(rep_dim, hidden_dim)
    
    if isinstance(envs.single_action_space, gym.spaces.Discrete)
        action_num = envs.single_action_space.n
        actor = PolicyNetDis(rep_dim, hidden_dim, action_num)
        
    if isinstance(envs.single_action_space, gym.spaces.Box)
        action_shape = envs.single_action_space
        action_dim = action_shape[1]
        actor = PolicyNetCont(rep_dim, hidden_dim, action_dim)
        
    # create agent
    lr = args.lr
    
    agent = PPOAgent(encoder, critic, actor).to(device)
    print('------ agent', agent)
    
    optimizer = torch.optim.Adam(list(agent.actor.parameters())+
                                 list(agent.critic.parameters())+
                                 list(agent.encoder.parameters()), 
                                 lr=lr, 
                                 eps=1e-5)
    
    # create data buffer
    num_steps = args.num_steps 
    num_envs = args.num_envs
    obs_space = envs.single_observation_space.shape
    action_space = envs.single_action_space.shape
    
    obs = torch.zeros((num_steps, num_envs) + obs_space).to(device)
    actions = torch.zeros((num_steps, num_envs) + action_space).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    # loops of collect data and update agent 
    episode_num = args.episode_num 
    num_steps = args.num_steps
    gamma = args.gamma 
    gae_lambda = args.gae_lambda
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    batch_size = int(num_envs*num_steps)
    minibatch_size = args.minibatch_size
    clip_coef = args.clip_coef
    clip_vloss = args.clip_vloss # True or False
    ent_coef = args.ent_coef 
    vf_coef = args.vf_coef
    max_grad_norm = args.max_grad_norm
    target_kl = args.target_kl

    for episode in tqdm(range(episode_num)):
        # reset the env 
        next_obs, _ = envs.reset(seed=seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(num_envs).to(device)
        
        # run one episode and collect data
        for step in range(0, args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch data of one episode and num_envs
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Train Agent
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(epoch_num):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
