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

        s_true = memory['paths_true'][rank].shape
        s_embedded = memory['paths_embedded'][rank].shape
        print('A',s_true)
        print('B',s_embedded)
        return memory['paths_true'][rank].reshape((1,*s_true,1)), memory['paths_embedded'][rank].reshape((1,*s_embedded,1))

strategies = {
    'best_reward' : best_reward,
}