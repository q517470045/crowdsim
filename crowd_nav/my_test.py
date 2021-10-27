import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=True, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--mixed', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    env_config_file = args.env_config
    policy_config_file = args.env_config

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    if args.mixed:
        env.test_sim = 'mixed'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0.2
        else:
            robot.policy.safety_space = 0

    policy.set_env(env)
    if args.visualize:
        for goalx in [-2, -1, 0, 1, 2]:
            for hgoalx in [-2, -1, 0, 1, 2]:
                ob = env.reset(goalx, 4, hgoalx, -4)
                done = False
                while not done:
                    action = robot.act(ob)
                    ob, done, info = env.step(action)
                if args.traj:
                    env.render('traj', args.video_file)
                else:
                    env.render('video', args.video_file)


if __name__ == '__main__':
    main()
