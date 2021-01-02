from gym.envs.registration import register

register(id='docking-v0',
         entry_point='gym_docking.envs:DockingEnv',
         )

register(id='hovering-v0',
         entry_point='gym_docking.envs:HoveringEnv',
         )