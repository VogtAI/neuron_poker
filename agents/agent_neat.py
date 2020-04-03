"""Player based on a trained neural network"""
# pylint: disable=wrong-import-order
import logging
import time

import numpy as np

from gym_env.env import Action

import json

import neat

log = logging.getLogger(__name__)



class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='NEAT', genome=None, config=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.genome = genome
        self.config = config
        self.env = env

        nb_actions = self.env.action_space.n

        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)


        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        #memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        #policy = TrumpPolicy()

        nb_actions = env.action_space.n


    @staticmethod
    def choose_similar_action(action, legal_actions):

        if action == Action.FOLD:
            return Action.CHECK
        if action == Action.CHECK:
            return Action.FOLD
        if action == Action.CALL:
            return Action.CHECK
        #if action == Action.RAISE_POT or action == Action.RAISE_HALF_POT or action == Action.RAISE_2POT:
        #  return Action.CALL # todo
        #print('error could not choose', action)
        return legal_actions[0]

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        #_ = observation  # not using the observation for random decision
        #_ = info


        #_ = this_player_action_space.intersection(set(action_space))


        actions = self.net.activate(observation)
        this_player_action_space = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                            Action.RAISE_2POT, Action.ALL_IN]
        self.env._get_legal_moves()
        legal_actions = self.env.legal_moves
        action = this_player_action_space[np.argmax(actions)]
        #print('legal ', legal_actions, ' ', action)
        if action not in legal_actions:
            action = self.choose_similar_action(action, legal_actions)

        return action


