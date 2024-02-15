import multiprocessing
from datetime import datetime
from logger_config import get_logger


def get_worker_number():
    try:
        worker = int(multiprocessing.current_process().name.split('-')[1])
    except:
        worker = 0
    return worker


def initialize(hardware_parameters,BASE_LOGS_FOLDER, DEBUG, DELAY, **KwargDict):
    import os
    import time
    if DEBUG:
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    time.sleep(DELAY)
    print('WORKER INITIALIZATION')
    # GET WORKER NUMBER
    worker = get_worker_number()
    logger = get_logger(name='worker-%s' % worker, filename=os.path.join(BASE_LOGS_FOLDER, 'worker-%s.log' % worker),
                        filemode='w')
    logger.info('Worker %s Initialization...' % worker)

    logger.info('Configuring tensorflow...')

    gpu_memory_limit = hardware_parameters['GPU_MEMORY']//hardware_parameters['GPU_WORKER']
    # CONFIGURE TENSORFLOW
    import tensorflow as tf
    # if worker <= hardware_parameters['GPU_WORKER']:
    GPU = 0
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.set_logical_device_configuration(
        tf.config.list_physical_devices('GPU')[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)])


    logger.info('Tensorflow configuration done!')


