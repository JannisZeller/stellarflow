import numpy as np

from abc import ABC, abstractmethod


class RewardHanlder(ABC):
    def set_reward_factors(
            self,
            reward_factor_boost,
            reward_factor_target_distance,
            reward_per_step,
            reward_per_step_near_target,
            near_target_window,
        ):
        self.reward_factor_boost = reward_factor_boost
        self.reward_factor_target_distance = reward_factor_target_distance
        self.reward_per_step = reward_per_step
        self.reward_per_step_near_target = reward_per_step_near_target
        self.near_target_window = near_target_window


    @abstractmethod
    def compute_reward(self):
        pass




class DistanceAndTargetReached(RewardHanlder):
    def compute_reward(self):
        distance = np.linalg.norm(self.target - self.walker.state_vector[:3])
        boost = self.current_boost[3:]
        reward = np.exp(
            + self.reward_per_step
            + self.reward_factor_boost * np.linalg.norm(boost)
            + self.reward_factor_target_distance * distance
        )
        if distance < self.near_target_window:
            reward += self.reward_per_step_near_target
        return reward


class ContinuousDistance(RewardHanlder):
    def compute_reward(self):
        distance = np.linalg.norm(self.target - self.walker.state_vector[:3])
        boost = self.current_boost[3:]
        reward = np.exp(
            + self.reward_per_step
            + self.reward_factor_boost * np.linalg.norm(boost)
            + self.reward_factor_target_distance * distance
        )
        return reward


class TargetReached(RewardHanlder):
    def compute_reward(self):
        distance = np.linalg.norm(self.target - self.walker.state_vector[:3])
        if distance < self.near_target_window:
            return self.reward_per_step_near_target
        return 0.
