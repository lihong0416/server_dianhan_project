import os
import logging
import logging.handlers


def get_logger_base_logging(is_save=False,
                            is_stream=True,
                            log_dir=None,
                            module_name='root',
                            log_level='INFO',
                            is_set_format=True,
                            when='midnight'):
    '''
    !!! 存在日志备份报错Bug !!!
    :param is_save:
    :param is_stream:
    :param log_dir:
    :param module_name:
    :param log_level:
    :param is_set_format:
    :param when:
    :return:
    '''
    logging.basicConfig()
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    if is_set_format:
        # set logging format
        format = '%(asctime)s-[%(filename)s, %(lineno)s]-%(levelname)s: %(message)s'
        # format = '%(asctime)s-%(levelname)s: %(message)s'
        formatter = logging.Formatter(format)

    if is_save:
        time_file_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, module_name + '.log'),
            when=when,
            # "midnight"：Roll over at midnight，"W"：Week day（0 = Monday），"D"：Days 天，"H"：Hour 小时，"M"：Minutes 分钟，"S"：Second 秒
            interval=1,
            backupCount=365  # 保留365天
        )
        time_file_handler.suffix = '%Y-%m-%d_%H-%M-%S.log'  # 按 秒
        if is_set_format:
            time_file_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(time_file_handler)

    if is_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(stream_handler)

    return logger


import datetime
import os.path


def get_yyyymmdd_str(format_str='%Y-%m-%d'):
    return format(datetime.datetime.now(), format_str)


class MyLogger:
    def __init__(self, module_name='root', is_save=True, is_stream=True, log_dir=None, encoding='gbk'):
        self.log_dir = log_dir
        self.is_save = is_save
        self.is_stream = is_stream
        self.module_name = module_name
        self.encoding = encoding

        if self.log_dir != None and isinstance(self.log_dir,str) and not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    def write(self, prt_content):
        if self.is_stream:
            print(prt_content)

        if self.is_save:
            new_name = f'{self.module_name}.{get_yyyymmdd_str()}.log'
            new_path = os.path.join(self.log_dir, new_name)
            with open(new_path, 'a', encoding=self.encoding) as f:
                f.write(prt_content)

    def info(self, content):
        prt_content = f'INFO-{datetime.datetime.now()}: {content}\n'
        self.write(prt_content)

    def error(self, content):
        prt_content = f'ERROR-{datetime.datetime.now()}: {content}\n'
        self.write(prt_content)

    def debug(self, content):
        prt_content = f'DEBUG-{datetime.datetime.now()}: {content}\n'
        self.write(prt_content)


def get_logger(is_save=False,
               is_stream=True,
               log_dir=None,
               module_name='root',
               log_level='INFO',
               is_set_format=True,
               when='midnight'):
    logger = MyLogger(module_name=module_name, is_save=is_save, is_stream=is_stream, log_dir=log_dir)
    return logger

# if __name__ == '__main__':
#     import time
#
#     log_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'logs'
#
#     if not os.path.isdir(log_dir):
#         os.makedirs(log_dir)
#     logger = get_logger(log_dir=log_dir, is_save=False)
#
#     for i in range(1, 100000):
#         time.sleep(1)
#         logger.info('hello')
