import itertools
from typing import Optional

import numpy as np
from numpy.random import Generator, default_rng


class GeneticAlgorithm:
    """
    A class to implement a basic Genetic Algorithm for optimization.

    :param action_bounds: A list of lists indicating the lower and upper bounds for each action.
    :param n_bits_per_action: An integer specifying the number of bits to represent an action.
    :param len_horizon: An integer indicating the length of the horizon for the optimization.
    :param n_iter: An integer indicating the number of iterations (generations) to run the algorithm.
    :param n_pop: An integer indicating the number of individuals (population size) in each generation.
    :param r_cross: A float indicating the crossover rate.
    :param r_mut: A float indicating the mutation rate.
    :param pop_init_actions: A list of actions to initialize the population for each optimization (e.g. from RL policy)
    :param allow_battery_idle: Shift battery action space bounds [-1,1] to allow a=0 at cost of reduced max. discharge.
    :param rng: An optional Generator instance for random number generation.
    :param verbose: An integer (0 or 1) indicating the verbosity level.
    """
    def __init__(
            self,
            action_bounds: list,
            n_bits_per_action: int,
            len_horizon: int,
            n_iter: int,
            n_pop: int,
            r_cross: float,
            r_mut: float,
            pop_init_actions: Optional[list] = None,
            allow_battery_idle: bool = False,
            rng: Optional[Generator] = None,
            verbose=0
    ):
        """
        Initialize the GeneticAlgorithm class.

        :param action_bounds: A list of lists indicating the lower and upper bounds for each action.
        :param n_bits_per_action: An integer specifying the number of bits to represent an action.
        :param len_horizon: An integer indicating the length of the horizon for the optimization.
        :param n_iter: An integer indicating the number of iterations (generations) to run the algorithm.
        :param n_pop: An integer indicating the number of individuals (population size) in each generation.
        :param r_cross: A float indicating the crossover rate.
        :param r_mut: A float indicating the mutation rate.
        :param pop_init_actions: A list of actions to initialize the population for each optimization
                                 (e.g. from RL policy)
        :param allow_battery_idle: Shift battery action space bounds [-1,1] to allow a=0
                                    at the cost of reduced max. discharge.
        :param rng: An optional Generator instance for random number generation.
        :param verbose: An integer (0 or 1) indicating the verbosity level.
        """
        self.bounds = action_bounds
        self.n_bits_per_action = n_bits_per_action
        self.len_horizon = len_horizon
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.pop_init_actions = pop_init_actions
        self.allow_battery_idle = allow_battery_idle
        self.rng = rng if rng is not None else default_rng(seed=None)
        self.verbose = verbose

        self.simulation_done = False

        # Compute total number of bits to encode action space
        self.n_actions = len(self.bounds[0])
        self.n_bits = self.n_actions * self.n_bits_per_action

    def decode(self, bitstring):
        """
        Decodes a bitstring.

        :param bitstring: A string that represents the bitstring to be decoded.
        :return: A list of decoded values.
        """
        decoded = []
        n_bits = self.n_bits
        largest = 2 ** (n_bits // self.n_actions) - 1
        for i in range(self.len_horizon):
            actions = []
            bit_start, bit_end = i * n_bits, (i + 1) * n_bits
            substring = bitstring[bit_start:bit_end]
            for j in range(self.n_actions):
                subbit_start, subbit_end = j * self.n_bits_per_action, (j + 1) * self.n_bits_per_action
                action_substring = substring[subbit_start: subbit_end]
                chars = ''.join([str(s) for s in action_substring])
                integer = int(chars, 2)

                lb = self.bounds[0][j]  # lower bound of action dimension
                ub = self.bounds[1][j]  # upper bound of action dimension

                # Convert the integer to the corresponding action using the bounds
                if (lb, ub) == (-1.0, 1.0) and self.allow_battery_idle:
                    # The +1 is to allow a=0 for battery
                    value = lb + (integer / (largest + 1)) * (ub - lb)
                else:
                    value = lb + (integer / largest) * (ub - lb)

                actions.append(value)
            decoded.append(actions)
        return decoded

    def selection(self, pop, scores, k=3):
        """
        Performs tournament selection.

        :param pop: A list that represents the population.
        :param scores: A list that represents the scores of the population.
        :param k: An integer that represents the number of 'healthy' parents.
        :return: A list of selected parents.
        """
        selection_ix = self.rng.integers(len(pop))
        for ix in self.rng.integers(low=0, high=len(pop), size=k - 1):
            if scores[ix] > scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def crossover(self, p1, p2):
        """
        Crossovers two parents to create two children.

        :param p1: A list that represents the first parent.
        :param p2: A list that represents the second parent.
        :return: A list of two children.
        """
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() < self.r_cross:
            pt = self.rng.integers(low=1, high=len(p1) - 2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(self, bitstring):
        """
        Performs a mutation on a bitstring.

        :param bitstring: A list that represents the bitstring to be mutated.
        """
        for i in range(len(bitstring)):
            if self.rng.random() < self.r_mut:
                bitstring[i] = 1 - bitstring[i]

    def get_env_score(self, env, actions):
        """
        Calculates the reward sum for a sequence of actions.

        :param env: An environment object.
        :param actions: A list that represents the sequence of actions.
        :return: A float that represents the reward sum.
        """
        r_sum = 0
        for a in actions:
            o, r, d, t, _ = env.step(np.array(a, dtype=np.float32))
            r_sum += r
            if d:
                self.simulation_done = True
        env.partial_reset(len(actions))
        return r_sum

    def run(self, env):
        """
        Performs the optimization of with the genetic algorithm.

        :param env: An environment object.
        :return: A tuple containing the best solution, the best evaluation, and if the end of episode was reached.
        """
        # Initialize population with specific actions, e.g. from RL agent
        if self.pop_init_actions is not None:
            encoder = BitEncoder(env_bounds=self.bounds,
                                 num_bits=self.n_bits,
                                 allow_battery_idle=self.allow_battery_idle)
            rl_pop = encoder.get_closest_bit(self.pop_init_actions[env.count:env.count + self.len_horizon])
            pop = [rl_pop for _ in range(self.n_pop)]
        # Randomly initialize population
        else:
            pop = [self.rng.integers(low=0, high=2, size=self.n_bits * self.len_horizon).tolist()
                   for _ in range(self.n_pop)]
        best, best_eval = 0, self.get_env_score(env, self.decode(pop[0]))

        for gen in range(self.n_iter):
            decoded = [self.decode(p) for p in pop]
            scores = [self.get_env_score(env, d) for d in decoded]

            for i in range(self.n_pop):
                if scores[i] >= best_eval:
                    best, best_eval = decoded[i], scores[i]
                    if self.verbose == 1:
                        print(">Generation %d, new best f(%s) = %.3f" % (gen, decoded[i], scores[i]))

            selected = [self.selection(pop, scores) for _ in range(self.n_pop)]
            children = list()
            for i in range(0, self.n_pop, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in self.crossover(p1, p2):
                    self.mutation(c)
                    children.append(c)
            pop = children

        return best, best_eval, self.simulation_done


class BitEncoder:
    """
    Encodes actions from a continuous space into a discrete bit representation based on specified environmental bounds
    and number of bits.

    This class is useful for scenarios where an algorithm requires discrete actions but the environment naturally
    operates in continuous action spaces. By encoding actions into bits, it bridges continuous action spaces with
    algorithms designed for discrete spaces.

    :param env_bounds: A list of two lists, with the first list containing the minimum values and the second list
                       containing the maximum values for each action dimension.
    :param num_bits: The total number of bits used for encoding all action dimensions.
    :param allow_battery_idle: Shift battery action space bounds [-1,1] to allow a=0 at cost of reduced max. discharge.
    """
    def __init__(
            self,
            env_bounds: list,
            num_bits: int,
            allow_battery_idle: bool = False,
    ):
        """
        Initializes the BitEncoder with environment bounds and the desired number of bits for encoding.

        :param env_bounds: A list of two lists, with the first list containing the minimum values and the second list
                           containing the maximum values for each action dimension.
        :param num_bits: The total number of bits used for encoding all action dimensions.
        :param allow_battery_idle: Shift battery action space bounds [-1,1] to allow a=0
                            at cost of reduced max. discharge.
        """
        self.bits_per_action = int(num_bits / len(env_bounds[0]))

        self.actions = []  # stores for each action dimension which actions are possible with chosen number of bits
        for a in range(len(env_bounds[0])):
            lb = int(env_bounds[0][a])  # lower bound of action dimension
            ub = int(env_bounds[1][a])  # upper bound of action dimension
            n_actions = 2 ** self.bits_per_action

            # Convert the integer to the corresponding action using the bounds
            if (lb, ub) == (-1, 1) and allow_battery_idle:
                action = np.linspace(lb, lb + ((n_actions-1)/n_actions) * (ub-lb), n_actions).tolist()
            else:
                action = np.linspace(lb, ub, n_actions).tolist()
            self.actions.append(action)

        self.bit_actions = [list(x) for x in itertools.product([0, 1], repeat=self.bits_per_action)]

    def get_closest_bit(self, act: list) -> list:
        """
        Encodes the provided action(s) into the closest corresponding bit representation based on the encoder
        configuration.

        :param act: A list of actions, where each action is represented as a list of values for each dimension.
        :return: A list of bits representing the closest encoded actions.
        """
        encoded_action = []
        for a in act:  # iterate over each action in the horizon
            for dim in range(len(a)):  # iterate over each action dimension
                closest_action = self._find_closest_action(a[dim], dim=dim)
                idx = self.actions[dim].index(closest_action)
                bit = self.bit_actions[idx]
                encoded_action.extend(bit)
        return encoded_action

    def _find_closest_action(self, a: float, dim: int) -> float:
        """
        Identifies the closest action in the predefined action space for a given continuous action value in a
        specified dimension.

        This method is utilized internally to determine the closest discrete action corresponding to a given
        continuous action.

        :param a: The continuous action value for which the closest discrete action is sought.
        :param dim: The dimension of the action for which the closest value is being determined.
        :return: The closest action value from the predefined action space.
        """
        return min(self.actions[dim], key=lambda y: abs(y - a))
