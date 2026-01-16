#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2023/3/14 14:59


import time
import json
import pymongo
import traceback
import os.path as osp
from threading import Thread

from ..utils.data_definition import defect_list
from ..utils.ob_utils import data_preprocess, segment_r_derivative, search_all_peaks, plot
from ..utils.strategy_utils import strategy_1

from sputils.logger_utils import get_logger
from sputils.read_rui import get_rui_from_dict
from sputils.url_utils import send_json_data


# 数据预处理进行的功能函数
def data_func(data_q, ob_q, idx, log_dir, sleep=0.002):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    logger = get_logger(is_save=True,
                        is_stream=True,
                        log_dir=log_dir,
                        module_name=f'data-process-{idx}',
                        log_level='INFO',
                        is_set_format=True,
                        when='midnight')
    while True:  # 死循环持续从data_q中获取数据
        try:
            if not data_q.empty():  # 判断队列是否不为空
                data = data_q.get()  # 获取数据
                logger.info(f"data process--Get id={data['id']} from data queue")
                rui = get_rui_from_dict(data['data'])
                # 对rui曲线进行数据预处理
                logger.info(f"data process--Data preprocess, id={data['id']}")
                is_weld2, r_sec_ori, r_sec, r_deriv_1, sp_sections_dict = data_preprocess(rui)
                if not is_weld2:
                    continue
                data['rui'] = rui
                data['r_sec'] = r_sec
                data['r_sec_ori'] = r_sec_ori
                data['r_deriv_1'] = r_deriv_1
                data['sp_sections_dict'] = sp_sections_dict
                # 矩阵放入rui_q队列
                ob_q.put(data)
                logger.info(f"data process--Put id={data['id']} into ob queue")
        except:
            logger.error(f"data process--{traceback.format_exc()}")
        time.sleep(sleep)  # 睡眠，减轻死循环造成的资源消耗


# 过烧计算进程的核心函数
def ob_func(ob_q, stt_1_q, idx, log_dir, sleep=0.002):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    logger = get_logger(is_save=True,
                        is_stream=True,
                        log_dir=log_dir,
                        module_name=f'ob-{idx}',
                        log_level='INFO',
                        is_set_format=True,
                        when='midnight')
    while True:  # 死循环持续从data_q中获取数据
        try:
            if not ob_q.empty():  # 判断队列是否不为空
                data = ob_q.get()  # 获取数据
                logger.info(f"ob process--Get id={data['id']} from ob queue")
                r_sec = data['r_sec']
                r_sec_ori = data['r_sec_ori']
                r_deriv_1 = data['r_deriv_1']

                deriv_r_secs = segment_r_derivative(r_sec, r_deriv_1)
                all_peaks = search_all_peaks(
                    r_sec_ori=r_sec_ori,
                    r_sec=r_sec,
                    deriv_r_secs=deriv_r_secs,
                )
                data['all_peaks'] = all_peaks
                stt_1_q.put(data)
                logger.info(f"ob process--Put id={data['id']} into plot queue")
        except:
            logger.error(f"ob process--{traceback.format_exc()}")
        time.sleep(sleep)  # 睡眠，减轻死循环造成的资源消耗


# 画图的核心函数
def plot_func(plot_q, idx, log_dir, save_dir, sleep=0.002):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    logger = get_logger(is_save=True,
                        is_stream=True,
                        log_dir=log_dir,
                        module_name=f'plot-{idx}',
                        log_level='INFO',
                        is_set_format=True,
                        when='midnight')
    while True:  # 死循环持续从data_q中获取数据
        try:
            if not plot_q.empty():  # 判断队列是否不为空
                data = plot_q.get()  # 获取数据
                logger.info(f"plot process--Get id={data['id']} from plot queue")
                rui = data['rui']
                all_peaks = data['all_peaks']
                sp_sections_dict = data['sp_sections_dict']
                plot(data, rui, all_peaks, sp_sections_dict, save_dir)
                logger.info(f"plot process--Plot, id={data['id']}")
                logger.info(f"plot process--Put id={data['id']} into send queue")
        except:
            logger.error(f"plot process--{traceback.format_exc()}")
        time.sleep(sleep)  # 睡眠，减轻死循环造成的资源消耗


# 发送数据
def send_func(send_q, idx, log_dir, send_result_list, db_url, db_name, col_name, sleep=0.002):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    logger = get_logger(is_save=True,
                        is_stream=True,
                        log_dir=log_dir,
                        module_name=f'send-data-{idx}',
                        log_level='INFO',
                        is_set_format=True,
                        when='midnight')
    # 创建MongoDB连接
    myclient = pymongo.MongoClient(db_url, directConnection=True, )
    mydb = myclient[db_name]
    cursor = mydb[col_name]
    while True:  # 死循环持续从data_q中获取数据
        try:
            if not send_q.empty():  # 判断队列是否不为空
                data = send_q.get()  # 获取数据
                logger.info(f"send process--Get id={data['id']} from plot queue")

                # 发送结果给所有相关链接
                for url_dict in send_result_list:
                    url = url_dict['url']
                    mode = url_dict['mode']
                    on_off = url_dict['on_off']
                    flag = int(url_dict['flag'])
                    json_data = {
                        'id': data['id'],
                        'result': data['result'],
                        'data': data['data'],
                        'flag': flag,
                    }
                    if int(on_off) != 1:  # on!=1, 不执行发送
                        continue
                    kwargs = {
                        'url': url,
                        'json_data': json_data,
                        'mode': mode,
                    }
                    t = Thread(target=send_json_data, kwargs=kwargs)
                    t.start()

                # 保存数据结果
                query = {'_id': data['data']['_id']}
                mongo_res = {
                    '$set': {
                        'Zero_WeldResult': data['result'],
                    }
                }
                cursor.update_one(query, mongo_res, upsert=True)
        except:
            logger.error(f"send process--{traceback.format_exc()}")
        time.sleep(sleep)  # 睡眠，减轻死循环造成的资源消耗


# 策略1函数
def strategy_1_func(
        stt_1_q,
        send_q,
        plot_q,
        log_dir,
        h_batch_thr,
        h_serious_thr,
        w_thr,
        batch_num,
        stat_dict_path,
        sleep=0.002
):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    logger = get_logger(is_save=True,
                        is_stream=True,
                        log_dir=log_dir,
                        module_name=f'stt1',
                        log_level='INFO',
                        is_set_format=True,
                        when='midnight')
    if not osp.isfile(stat_dict_path):
        stat_dict = {}
    else:
        _, extname = osp.splitext(stat_dict_path)
        if extname.upper() == "JSON":
            with open(stat_dict_path, 'r', encoding='utf8') as f:
                stat_dict = json.load(f)
        else:
            stat_dict = {}
    while True:  # 死循环持续从data_q中获取数据
        try:
            if not stt_1_q.empty():  # 判断队列是否不为空
                data = stt_1_q.get()  # 获取数据
                logger.info(f"strategy-1 process--Get id={data['id']} from stt_1 queue")
                ori_data = data['data']
                all_peaks = data['all_peaks']
                ob_code, stat_dict = strategy_1(
                    data=ori_data,
                    all_peaks=all_peaks,
                    h_batch_thr=h_batch_thr,
                    h_serious_thr=h_serious_thr,
                    w_thr=w_thr,
                    batch_num=batch_num,
                    stat_dict=stat_dict,
                    stat_dict_path=stat_dict_path
                )

                if ob_code == 0:
                    result = '合格'
                else:
                    result = '过烧'
                data['ob_code'] = ob_code
                data['result'] = result

                if ob_code == 1:
                    batch_serious = 'batch'
                elif ob_code == 2:
                    batch_serious = 'serious'
                else:
                    batch_serious = 'normal'
                data['batch_serious'] = batch_serious

                if ob_code > 0:
                    logger.info(
                        f"strategy-1 process--Find SP Overburn! id={data['id']}, this defect is {batch_serious}")
                    plot_q.put(data)
                    logger.info(f"strategy-1 process--Put id={data['id']} into plot queue")
                send_q.put(data)
                logger.info(f"strategy-1 process--Put id={data['id']} into send queue")
        except:
            logger.error(f"plot process--{traceback.format_exc()}")
        time.sleep(sleep)  # 睡眠，减轻死循环造成的资源消耗
