import numpy as np
import torch
import gym
import argparse
import os
import wandb
import tqdm
import subprocess
import matplotlib.pyplot as plt
import time


from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path

from agents.our_ddpg.utils import ReplayBuffer
from agents.our_ddpg.preprocessor import Preprocessor

from agents.our_ddpg.DASMRNOE import DDPG
from agents.our_ddpg.loggers import Logger

from exceptions import AlreadyRunningException
from wrappers import EnvWrapper


def objective(numstations):
    # Hyperparameter search space
    hidden_sizes = [178, 270]
    actor_lr = 4.972e-5
    critic_lr = 9.077e-5
    tau = 0.003188611
    policy_noise = 0.350021220
    noise_clip = 0.289696247
    discount = 0.822717174
    q_weight = 0.166264636
    regularization_weight = 0.005807109
    smr_ratio = 10

    # check scenario.h to see and edit the scenarios
    scenario = "basic"  # convergence #basic
    agent_being_trained = "OurDDPG"

    simTime = 15
    stepTime = 0.01
    history_length = 300
    steps_per_ep = int(simTime/stepTime)
    EPISODE_COUNT = 12

    nWifi = numstations

    sim_args = {
        "simTime": simTime,
        "envStepTime": stepTime,
        "historyLength": history_length,
        "agentType": "continuous",
        "scenario": scenario,
        "nWifi": nWifi,
    }
    tags = ["Rew: normalized speed",
            "OurDDPG",
            sim_args['scenario'],
            f"Actor: UNDEFINED",
            f"Critic: UNDEFINED",
            f"Instances: 1",
            f"Station count: {sim_args['nWifi']}",
            *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

    wtags = [f"{EPISODE_COUNT}ep training", f"{simTime}s", f"{nWifi} nWifi", f"envStep {stepTime}", agent_being_trained, "train"]

    run = wandb.init(project="my-awesome-project", tags = wtags, name="DASMR-CW20-STA"+str(nWifi)+"-HYP""basic""noexplor")

    logger = Logger(False, tags, None, experiment=None)
    logger.begin_logging(EPISODE_COUNT, steps_per_ep, None, None, stepTime)
    preprocess = Preprocessor(False).preprocess

    print("Steps per episode:", steps_per_ep)

    threads_no = 1
    env = EnvWrapper(threads_no, **sim_args)
    env.reset()


    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="OurDDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="ContentionWindow")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=300, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=10e10, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1) #0.1         # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int) # 256      # Batch size for both actor and critic
    parser.add_argument("--discount", default=discount) #0.99             # Discount factor
    parser.add_argument("--tau", default=tau) # 0.005                    # Target network update rate
    parser.add_argument("--policy_noise", default=policy_noise) # R:TD3 only              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=noise_clip) # R:TD3 only                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int) # R:TD3 only       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 1
    action_dim = 1
    max_action = 1  # DDPG outputs action from -1 to 1
    real_max_action = 6  # actions is scaled to be in range from 0 to 6
    stateSize = state_dim


    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "hidden_sizes": hidden_sizes
    }

    # Initialize policy
    if args.policy == "OurDDPG":
        policy = DDPG(**kwargs)
    else:
        print("Policy not available")
    
    # Update optimizer learning rates
    policy.actor1_optimizer = torch.optim.Adam(policy.actor1.parameters(), lr=actor_lr)
    policy.actor2_optimizer = torch.optim.Adam(policy.actor2.parameters(), lr=actor_lr)
    policy.critic1_optimizer = torch.optim.Adam(policy.critic1.parameters(), lr=critic_lr)
    policy.critic2_optimizer = torch.optim.Adam(policy.critic2.parameters(), lr=critic_lr)
    
    # Set additional hyperparameters
    policy.policy_noise = policy_noise
    policy.noise_clip = noise_clip
    policy.q_weight = q_weight
    policy.regularization_weight = regularization_weight
    policy.smr_ratio = smr_ratio

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    final_avg_throughput = 0.0
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    time_step = 0
    for episode in range(EPISODE_COUNT):
        print("Episode:", episode+1)
        episode_reward = 0

        try:
            env.run()
        except AlreadyRunningException as e:
            pass

        avg_throughput = 0.0
        sent_mb = 0
        obs_dim = 1
        state = env.reset()
        state = state[0][:stateSize]
        state = np.reshape(state, stateSize)
        last_action = None
        with tqdm.trange(1, steps_per_ep+1) as t:
            for step in t: #

                # Select action randomly or according to policy
                if time_step < args.start_timesteps:
                    action = np.array([[np.random.uniform(0,6)]])
                    real_action = action
                else:
                    action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)
                    real_action = real_max_action * (action + 1) / 2   # scale it from -1,1 to 0,6

                    real_action = np.array([real_action])

                # Perform action
                next_state, reward, done, info = env.step(real_action)  # inserting action between 0 and 6
                not_processed_state = preprocess(np.reshape(next_state, (-1, len(env.envs), obs_dim)))

                next_state = next_state[0][:stateSize]
                next_state = np.reshape(next_state, stateSize)


                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done)  # inserting action between -1 and 1
                

                state = next_state
                episode_reward += reward
                if step > 300:
                    actor_loss = policy.actor_loss
                    critic_loss = policy.critic_loss
                    if(isinstance(policy.actor_loss, torch.Tensor)):
                        actor_loss = actor_loss.detach().cpu().numpy()
                    if(isinstance(policy.critic_loss, torch.Tensor)):
                        critic_loss = critic_loss.detach().cpu().numpy()
                    loss = {"actor_loss": actor_loss, "critic_loss": critic_loss}
                    logger.log_round(state, reward, episode_reward, info, loss, np.mean(not_processed_state, axis=0)[0], episode*steps_per_ep+step)
                    avg_throughput += logger.current_speed

                t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")


                # Train agent after collecting sufficient data
                if time_step >= args.start_timesteps:
                    policy.train(replay_buffer, args.batch_size)

                time_step += 1

                if done:
                    print(f"Total T: {t+1} Episode Num: {episode} Episode T: {t} Reward: {episode_reward:.3f}")

                    break
        logger.log_episode(episode_reward, logger.sent_mb/(simTime), episode)
        avg_throughput /= steps_per_ep
        print("Average Throughput:", avg_throughput)
        final_avg_throughput += avg_throughput
        env.close()
    run.finish()
    final_avg_throughput /= 12
    print("Final Average Throughput:", final_avg_throughput)
    return final_avg_throughput
   
for i in range(50, 51, 5):
    objective(i)

	
