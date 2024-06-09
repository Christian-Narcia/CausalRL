# base code from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
import argparse
import os
import random
import time
from distutils.util import strtobool
from itertools import permutations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN
from torch_geometric.nn import aggr
from torch_geometric.nn import LayerNorm


# continuous action
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

#MultiProcess
import concurrent.futures


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# descrete action
class AgentD(nn.Module):
    def __init__(self, envs):
        super(AgentD, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # unnormalized action probabilites
        logits = self.actor(x)
        # Catigorial distribution (essensially a soft max operation to get action probablility)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
# Continuous action
class AgentC(nn.Module):
    def __init__(self, envs, nodesNum, edge_index=()):
        super(AgentC, self).__init__()
        self.critic = GraphCritic(nodesNum, edge_index=edge_index)
        self.actor_mean = GraphActor(nodesNum, 7, edge_index=edge_index)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class GraphActor(nn.Module):
    def __init__(self, nodesNum, action_space=7, hidden_dim=64, edge_index=()):
        super().__init__()
        self.conv1 = GCNConv(-1, nodesNum) # set to 1 timestep observations
        self.layer_norm = LayerNorm(nodesNum)
        self.aggr = aggr.MedianAggregation()
        self.edge_index = edge_index

        # original layers
        self.origLin1 = layer_init( nn.Linear(nodesNum, hidden_dim) )
        self.origLin2 = layer_init( nn.Linear(hidden_dim, hidden_dim) )
        self.origLin3 = layer_init( nn.Linear(hidden_dim, action_space) )

    def forward(self, x):
        # print(x, self.edge_index)
        x = torch.reshape(x, (-1, 23, 1))
        x = self.conv1(x, self.edge_index)
        x = self.aggr(x)
        x = F.tanh( self.origLin1(x) )
        x = F.tanh( self.origLin2(x) )
        # out = self.origLin3(x)
        out = torch.reshape(self.origLin3(x), (-1, 7))
        return out

class GraphCritic(nn.Module):
    def __init__(self, nodesNum, hidden_dim=64, edge_index=()):
        super().__init__()
        self.conv1 = GCNConv(-1, nodesNum)
        self.layer_norm = LayerNorm(nodesNum)
        self.aggr = aggr.MedianAggregation()
        self.edge_index = edge_index

        # original layers
        self.origLin1 = layer_init( nn.Linear(nodesNum, hidden_dim) )
        self.origLin2 = layer_init( nn.Linear(hidden_dim, hidden_dim) )
        self.origLin3 = layer_init( nn.Linear(hidden_dim, 1), std=1.0 )        

    def forward(self, x):
        # print(x, self.edge_index)
        x = torch.reshape(x, (-1, 23, 1))
        x = self.conv1(x, self.edge_index)
        x = self.aggr(x)
        x = F.tanh( self.origLin1(x) )
        x = F.tanh( self.origLin2(x) )
        # out = self.origLin3(x)        
        out = torch.reshape(self.origLin3(x), (-1, 1))
        return out

def graph_nodes(edgeList):
    import matplotlib.pyplot as plt
    from torch_geometric.utils.convert import to_networkx
    import networkx as nx
    from torch_geometric.data import Data
    node_labels = ['shX', 'shY', 'shZ', 'elX', 'elY', 'wrX', 'wrY', 'shVx', 'shVy', 'shVz', 
                   'elVx', 'elVy', 'wrVx', 'wrVy', 'fX', 'fY', 'fZ', 'oX', 'oY', 'oZ', 'gX', 'gY', 'gZ']
    node_labels = {i:node_labels[i] for i in range(len(node_labels))}
    zero_array = np.zeros((23, 1))
    data = Data(x=zero_array, edge_index=edgeList)
    print(data)
    netGraph = to_networkx(data, to_undirected=True)
    
    nx.draw(netGraph, with_labels=True, node_size=500, node_color='lightblue', font_size=18, labels=node_labels)
    plt.show()


# Graph construction from two timesteps of observations
def create_torch_graph_data(typeGNN, nodeNum):
    '''
    Pusher observation order is:
        0 - shX     (sh is shoulder)
        1 - shY 
        2 - shZ
        3 - elX     (el is elbow)
        4 - elY
        5 - wrX     (wr is wrist)
        6 - wrY
        7 - shVx    (V is velocity)
        8 - shVy
        9 - shVz
        10 - elVx
        11 - elVy
        12 - wrVx
        13 - wrVy
        14 - fX     (f is finger)
        15 - fY
        16 - fZ
        17 - oX     (o is object to be moved)
        18 - oY
        19 - oZ
        20 - gX     (g is goal)
        21 - gY
        22 - gZ
    '''
    match typeGNN:
        case 0:
            return [(i, i) for i in range(nodeNum)]
        case 1:
            return list(permutations([i for i in range(nodeNum)], 2))
        case 2:
            return [(5,7),(20,10),(18,12),(3,6),(4,11),(9,11),(5,7),(17,22),(21,6),(4,2),(1,0),(10,12),(14,13),(7,3),(5,2),(14,13),
                    (17,5),(13,12),(9,3),(0,12),(0,22),(8,15),(15,17),(8,16),(3,19)]
        case 3:
            return [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,0),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,13)]
        case 4:
            return [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,0)]
        case _:
            return "Unknown value for 'graph_type'."
    
    # Fully Connected
    # edge_index = torch.tensor(edges, dtype=torch.long)
    # edge_index = edge_index.t().contiguous()

    # Empty, no connections
    # edge_index = torch.tensor([], dtype=torch.long)
    
    # New Empty list 
    # n = 44
    # edges = [(i, i) for i in range(n + 1)]
    # edge_index = torch.tensor(edges, dtype=torch.long).contiguous()

    # # concat all features for nodes from timestep t1 and t2
    # temporal_features = [[d[i]] for i in range(len(d))] + [[d2[i]] for i in range(len(d))]
        
    # node_feature = temporal_features

    # node_feature = torch.tensor(node_feature, dtype=torch.float)

    # data = Data(x=node_feature, edge_index=edge_index)

    # return data

def main(args, envName, discrete, seed, expNum):
    # learning_rate = 0.00025
    learning_rate = 0.0003
    total_timesteps = 500000

    # not implemented properly yet
    capture_video = False

    # Algorithm specific arguments
    ##############
    #the number of parallel game environments
    # num_envs = 4
    num_envs = 1
    #the number of steps to run in each environment per policy rollout
    # num_steps = 128
    num_steps = 2048
    #Toggle learning rate annealing for policy and value networks
    anneal_lr = True
    # Use GAE for advantage computation
    gae  = True
    #the discount factor gamma
    gamma = 0.99
    #the lambda for the general advantage estimation
    gae_lambda = 0.95
    #the number of mini-batches
    num_minibatches = 32
    #the K epochs to update the policy
    # update_epochs = 4
    update_epochs = 10
    #Toggles advantages normalization
    norm_adv = True
    #the surrogate clipping coefficient
    clip_coef = 0.2
    #Toggles whether or not to use a clipped loss for the value function, as per the paper.
    clip_vloss = True
    # coefficient of the entropy
    # ent_coef = 0.01
    ent_coef = 0.0
    #coefficient of the value function
    vf_coef = 0.5
    # the maximum norm for the gradient clipping
    max_grad_norm = 0.5
    # the target KL divergence threshold
    target_kl = None


    batch_size = int(num_envs * num_steps)

    minibatch_size = int(batch_size // num_minibatches)

    # Types of Graphs
    graph_types = {
    0: "Self-Connected",
    1: "FC",
    2: "Random",
    3: "2-Loops",
    4: "Circle"
    }
    graph_type = 4

    
    
    

    run_name = f"{envName}_GNN_1NormFalse_{graph_types[graph_type]}_{seed}__{expNum}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup 
    envs = gym.vector.SyncVectorEnv(
        [make_env(envName, seed + i, i, capture_video, run_name) for i in range(num_envs)]
    )
    if discrete:
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = AgentD(envs).to(device)
    else:
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        nodeNum = 23
        # print(create_torch_graph_data(graph_type,nodeNum))
        edges = torch.tensor(create_torch_graph_data(graph_type,nodeNum), dtype=torch.long).to(device)
        # edges = torch.tensor([(i, i) for i in range(23)], dtype=torch.long).to(device) #torch.tensor(list(permutations([i for i in range(46)], 2)), dtype=torch.long).to(device)
        EDGE_INDEX = edges.t().contiguous()
        graph_nodes(edges.t())
        print(EDGE_INDEX.shape)
        # exit()
        agent = AgentC(envs, nodeNum,EDGE_INDEX).to(device)

    # eps 1e-5 from orinial implementation
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    # store initial observation
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    #store initial termination condition to be false
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size
    
    # Training LOOP
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info= envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)


            for value_array in info.values():
                for item in value_array:
                    if isinstance(item, dict) and "episode" in item:
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # First foward pass of minibatch obs
                if discrete:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
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

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Help reproduce experiement?
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    
    # CUDA 
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    

    args = parser.parse_args()

    # envName = "CartPole-v1"
    # discrete = True

    envName = "Pusher-v4"
    discrete = False

    seed = 1 

    # Uncomment and comment out MULTIPROCESSING with for single run
    expNum = 1
    main(args, envName, discrete, seed, expNum)


    
    # FOR MULTIPROCESSING 
    # totalRuns = 1
    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     futures = [executor.submit(main, args, envName, discrete, seed, expNum) for expNum in range(totalRuns)]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()  # Retrieve result or propagate exception
    #         except Exception as e:
    #             print(f"An experiment failed with error: {e}")