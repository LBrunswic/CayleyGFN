from tensorboard.plugins.hparams import api as hp_api
import numpy as np
import itertools
def check_consistency(iter):
    iter = list(iter)
    if len(iter) == 0:
        return True
    T = type(iter[0])
    return all([isinstance(x,T) for x in iter])

def hp_compile(hp_value,hp_description):
    print(hp_value, hp_description)
    if hp_description == 'freefloat':
        return hp_api.RealInterval(min_value=-np.inf,max_value=np.inf)

    if hp_description == 'freeint':
        return hp_api.RealInterval(min_value=-np.inf,max_value=np.inf)

    if hp_description == 'freedict':
        return hp_api.Discrete([str(x) for x in hp_value])

    if hp_description == 'freebool':
        return hp_api.Discrete([True,False])

    if isinstance(hp_description,list):
        assert(check_consistency(hp_description))
        assert(all([x in hp_description for x in hp_value]))
        return hp_api.Discrete(hp_value)

    if isinstance(hp_description,tuple):
        assert(check_consistency(hp_description[0]))
        return hp_api.Discrete([hp_description[1](x) for x in hp_value])

    raise NotImplementedError


def record_possible_hp(hp_values,hp_descriptions):
    HP = {
        name: hp_api.HParam(name,hp_compile(hp_values[name],hp_descriptions[name]))
        for name in hp_values
    }
    return HP


def record_case_HP(log_dir,HP):
    HP = list(HP.values())
    import tensorflow as tf
    from TestingEnv.metrics import metrics
    with tf.summary.create_file_writer(log_dir).as_default():
        hp_api.hparams_config(
            hparams=HP,
            metrics=[hp_api.Metric(m, display_name=m) for m in ["loss"] + [m().name for m in metrics]],
        )


def build_hp_list(hparams):
    keys = list(hparams.keys())
    hparams_list = [
        {
            key : hp_set[i]
            for i, key in enumerate(keys)
        }
        for hp_set in list(itertools.product(*[hparams[key] for key in keys]))
    ]
    return hparams_list