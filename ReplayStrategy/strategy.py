import numpy as np


class best_reward:
    def __init__(self,seed=1234,delta=0.01):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.delta = delta
        self.name = 'best_reward'

    def HP(self):
        return {
            'strat_seed': self.seed,
            'strat_delta': self.delta,
            'strat_name': self.name
        }
    def __call__(self, memory, shape):

        print(np.sum(memory['paths_reward'], axis=1)+self.delta)
        print((np.sum(memory['paths_reward'], axis=1)+self.delta).shape)
        p=np.sum(memory['paths_reward'], axis=1)+self.delta
        p = p / np.sum(p)
        rank = self.rng.choice(
            a=np.arange(memory['paths_true'].shape[0]),
            size=shape[1],
            replace=False,
            p=p
        )


        return memory['paths_true'][rank].reshape(shape), memory['paths_embedded'][rank].reshape(shape)

strategies = {
    'best_reward' : best_reward,
}