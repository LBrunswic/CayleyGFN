import logging
import os
def get_logger(*args,filename=None,name=__name__,**kwargs):
    # logging.basicConfig(*args,**kwargs)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.FileHandler(filename, mode='a')
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # logger.debug('debug message')

    # logger.warning('warn message')
    # logger.error('error message')
    # logger.critical('critical message')
    # logging.basicConfig(filename=filename, encoding='utf-8', level=logging.DEBUG)
    return logger

def log_dict(logger,to_log,name='',level='debug',prefix=''):
    if isinstance(to_log,dict):
        logger.debug(name)
        for key in to_log:
            log_dict(logger,to_log[key],level=level,prefix=prefix+' ')
    elif isinstance(to_log,str):
        logger.debug(to_log)
    else:
        logger.debug(str(to_log))

def log_tensorlist(logger,to_log,name='',level='debug',prefix=''):
    logger.debug(name)
    for tensor in to_log:
        logger.debug(tensor.name)
        logger.debug(str(tensor))

def check_success(folders):
    worker_logs = []
    for folder in folders:
        worker_logs.append([os.path.join(folder,x) for x  in os.listdir(folder) if 'worker' in x][1:])
    for worker_log in worker_logs:
        for log in worker_log:
            with open(log,'r') as f:
                lines = f.readlines()
                if 'Experiment done' not in lines[-1]:
                    return False
