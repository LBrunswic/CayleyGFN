import os,time
from datetime import datetime
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

from TestingEnv import hyperparametersutils, poolutils
from Hyperparameters.Hyperparameter import MODEL_HYPERPARAMETERS, PROBLEM_HYPERPARAMETERS, naming, SEEDS
from multiprocessing import Pool
from logger_config import get_logger, log_dict
from Hyperparameters.agregate import EXPERIMENTAL_SETTINGS
timestamp = str(datetime.now())
BASE_LOGS_FOLDER = os.path.join('..', 'logs', 'CayleyGFN', timestamp)
BASE_DATA_FOLDER = 'data/'
os.makedirs(BASE_LOGS_FOLDER,exist_ok=True)
os.makedirs(BASE_DATA_FOLDER,exist_ok=True)
logger = get_logger(name='general',filename=os.path.join(BASE_LOGS_FOLDER,'general.log'), filemode='w')
logger.info('START')
TEST = False
if __name__ == '__main__':


    TO_DO = [
        'S15W1_light_noBeta',
        # 'S5W1_saniity_check'
    ]

    SERIES_NAME = ''+'TEST'*TEST

    for HP_CASE in TO_DO:

        FIXED_HYPERPARAMETERS = EXPERIMENTAL_SETTINGS[HP_CASE]['FIXED_HYPERPARAMETERS']
        TUNING_HYPERPARAMETERS = EXPERIMENTAL_SETTINGS[HP_CASE]['TUNING_HYPERPARAMETERS']
        DENSITY_PARAMETERS = EXPERIMENTAL_SETTINGS[HP_CASE]['DENSITY_PARAMETERS']
        HARDWARE_PARAMETERS = EXPERIMENTAL_SETTINGS[HP_CASE]['HARDWARE_PARAMETERS']
        logger.info('Hyperparameters %s' % HP_CASE)
        log_dict(logger,FIXED_HYPERPARAMETERS,name='FIXED_HYPERPARAMETERS')
        log_dict(logger,TUNING_HYPERPARAMETERS,name='TUNING_HYPERPARAMETERS')
        log_dict(logger,DENSITY_PARAMETERS,name='DENSITY_PARAMETERS')
        log_dict(logger,HARDWARE_PARAMETERS,name='HARDWARE_PARAMETERS')



        log_dir = os.path.join(BASE_LOGS_FOLDER, '_'.join([
            naming(FIXED_HYPERPARAMETERS, TUNING_HYPERPARAMETERS),
            SERIES_NAME,
        ]))

        HP = hyperparametersutils.record_possible_hp(
            {
                'seed': SEEDS[:DENSITY_PARAMETERS['SEED_REPEAT'][0]],
                **DENSITY_PARAMETERS,
                **FIXED_HYPERPARAMETERS,
                **TUNING_HYPERPARAMETERS
            },
            {**MODEL_HYPERPARAMETERS, **PROBLEM_HYPERPARAMETERS}
        )
        HP_DESCRIPTION = {**MODEL_HYPERPARAMETERS, **PROBLEM_HYPERPARAMETERS}
        hparams_list = hyperparametersutils.build_hp_list(
            {'seed': SEEDS[:DENSITY_PARAMETERS['SEED_REPEAT'][0]],**DENSITY_PARAMETERS, **FIXED_HYPERPARAMETERS, **TUNING_HYPERPARAMETERS}
        )
        for hparams in hparams_list:
            hparams.update({
                'logdir': log_dir,
                'pool_size': HARDWARE_PARAMETERS['POOL_SIZE']
            })

        def run_wrapper(experiments_hparams):

            from TestingEnv.poolutils import get_worker_number
            import os
            import logging
            worker = get_worker_number()
            logger = logging.getLogger(name='worker-%s' % worker)


            import os
            import tensorflow as tf
            from tensorboard.plugins.hparams import api as hp

            experiments_hparams['logdir'] = os.path.join(experiments_hparams['logdir'], str(worker)+'-' + str(datetime.now()))
            from train import train_test_model
            logger.info('Launch experiment...')

            result, returns,flow = train_test_model(experiments_hparams,logger)
            logger.info('Experiment done!')
            data_save = os.path.join(BASE_DATA_FOLDER,str(worker)+'-'+str(datetime.now())+'.csv')
            result.to_csv(data_save)
            hp_columns = list(experiments_hparams.keys()) + ['epoch','episode']
            res_columns = [column for column in result.columns if column not in hp_columns]
            hparam_dict = result[hp_columns].to_dict()
            for hyperparam_name in HP_DESCRIPTION:
                if isinstance(HP_DESCRIPTION[hyperparam_name], tuple):
                    hparam_dict[hyperparam_name] = [
                        HP_DESCRIPTION[hyperparam_name][1](hparam_dict[hyperparam_name][experiment])
                        for experiment in range(len(result))
                        ]
                if isinstance(HP_DESCRIPTION[hyperparam_name], str) and HP_DESCRIPTION[hyperparam_name] == 'freedict':
                    hparam_dict[hyperparam_name] = [
                        str(hparam_dict[hyperparam_name][experiment])
                        for experiment in range(len(result))
                        ]

            for experiment in range(len(result)):
                folder = experiments_hparams['logdir'] + '-%s' % experiment
                folder = folder.replace(' ','').replace(':','').replace('.','')
                with tf.summary.create_file_writer(folder).as_default():
                    hp.hparams({key:hparam_dict[key][experiment] for key in hparam_dict})
                    for value in res_columns:
                        tf.summary.scalar(value, result[value].values[experiment], step=1)
            return returns

        if TEST:
            import os
            import tensorflow as tf
            from tensorboard.plugins.hparams import api as hp
            from TestingEnv.poolutils import get_worker_number

            experiments_hparams=hparams_list[0]
            experiments_hparams['logdir'] = os.path.join(experiments_hparams['logdir'],
                                                         'TEST' + '-' + str(datetime.now()))
            from train import train_test_model

            result, returns,flow = train_test_model(experiments_hparams)

        else:
            with Pool(
                    processes=HARDWARE_PARAMETERS['GPU_WORKER']+HARDWARE_PARAMETERS['CPU_WORKER'],
                    initializer=poolutils.initialize,
                    initargs=(HARDWARE_PARAMETERS,BASE_LOGS_FOLDER)
            ) as pool:
                pool.apply(hyperparametersutils.record_case_HP, args=(log_dir, HP))
                results = pool.map(run_wrapper, hparams_list, chunksize=1)

    print("THAT'S ALL FOLKS")
