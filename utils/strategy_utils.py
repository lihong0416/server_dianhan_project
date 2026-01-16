import json
import os.path as osp
import pandas as pd

from sputils.datetime_utils import str_to_datetime, datetime_to_str


# 根据阈值判断结果是否

# 将阈值df转为字典
def get_thr_dict(thr_df):
    result = {}
    for i_r in range(thr_df.shape[0]):
        row = thr_df.iloc[i_r, :]
        id = row['id']
        thr = row['ob_thr']
        if id not in result:
            result[id] = thr
    return result


# 根据焊点获得阈值
def get_thr(sp_tag, thr_dict):
    return thr_dict[sp_tag]


# 策略1
def strategy_1(data,
               all_peaks,
               h_batch_thr=0.02,
               h_serious_thr=0.04,
               w_thr=10,
               batch_num=3,
               stat_dict=None,
               stat_dict_path=None):
    try:
        spot_tag = data['spot_tag']
    except:
        spot_tag = data['spotTag']
    dt_str = data['dateTime']
    dt_obj = str_to_datetime(dt_str)
    dt_str = datetime_to_str(dt_obj, format='%Y-%m-%d')
    key = f"{spot_tag}#{dt_str}"

    # 判断stat_dict是否为dict，如果不是，则从json文件stat_dict_path中加载dict
    if not isinstance(stat_dict, dict):
        if not osp.isfile(stat_dict_path):
            stat_dict = {}
        else:
            try:
                _, extname = osp.splitext(stat_dict_path)
                if extname.upper() == "JSON":
                    with open(stat_dict_path, 'r', encoding='utf8') as f:
                        stat_dict = json.load(f)
                else:
                    stat_dict = {}
            except:
                stat_dict = {}

    # 如果stat_dict中未记录key，则创建key-value
    if key not in stat_dict:
        stat_dict[key] = 0

    # 获取核心信息
    if len(all_peaks) == 0:
        return 0, stat_dict
    left_h = all_peaks[0][1]['left_h']
    box = all_peaks[0][1]['box']
    box_w = box[2] - box[0]

    # 如果最大山峰的宽度不足，则返回 0（表示正常）和stat_dict
    if box_w < w_thr:
        return 0, stat_dict
    # 判断使用“严重”策略还是使用“批量”策略
    if h_serious_thr > left_h >= h_batch_thr:  # 左峰高度在高度批量阈值和高度严重阈值之间，进行批量统计和判断
        stat_dict[key] += 1
        stat_count = len(list(stat_dict.values()))
        if stat_count > 0 and stat_count % 10:
            with open(stat_dict_path, 'w', encoding='utf8') as f:
                json.dump(stat_dict, f)
        if stat_dict[key] >= batch_num:  # 判断数量是否达到批量阈值
            return 1, stat_dict
        else:
            return 0, stat_dict
    elif left_h >= h_serious_thr:  # 左峰高度达到高度严重阈值
        stat_dict[key] += 1
        stat_count = len(list(stat_dict.values()))
        if stat_count > 0 and stat_count % 10:
            with open(stat_dict_path, 'w', encoding='utf8') as f:
                json.dump(stat_dict, f)
        return 2, stat_dict
    else:
        return 0, stat_dict

# 策略1
def strategy_1AEQ(data,
               all_peaks,
               h_batch_thr=0.02,
               h_serious_thr=0.04,
               w_thr=10,
               batch_num=3,
               stat_dict=None,
               stat_dict_path=None):
    try:
        spot_tag = data['spot_tag']
    except:
        spot_tag = data['spotTag']
    dt_str = data['rui_time']
    dt_obj = str_to_datetime(dt_str)
    dt_str = datetime_to_str(dt_obj, format='%Y-%m-%d')
    key = f"{spot_tag}#{dt_str}"

    # 判断stat_dict是否为dict，如果不是，则从json文件stat_dict_path中加载dict
    if not isinstance(stat_dict, dict):
        if not osp.isfile(stat_dict_path):
            stat_dict = {}
        else:
            try:
                _, extname = osp.splitext(stat_dict_path)
                if extname.upper() == "JSON":
                    with open(stat_dict_path, 'r', encoding='utf8') as f:
                        stat_dict = json.load(f)
                else:
                    stat_dict = {}
            except:
                stat_dict = {}

    # 如果stat_dict中未记录key，则创建key-value
    if key not in stat_dict:
        stat_dict[key] = 0

    # 获取核心信息
    if len(all_peaks) == 0:
        return 0, stat_dict
    left_h = all_peaks[0][1]['left_h']
    box = all_peaks[0][1]['box']
    box_w = box[2] - box[0]

    # 如果最大山峰的宽度不足，则返回 0（表示正常）和stat_dict
    if box_w < w_thr:
        return 0, stat_dict
    # 判断使用“严重”策略还是使用“批量”策略
    if h_serious_thr > left_h >= h_batch_thr:  # 左峰高度在高度批量阈值和高度严重阈值之间，进行批量统计和判断
        stat_dict[key] += 1
        stat_count = len(list(stat_dict.values()))
        if stat_count > 0 and stat_count % 10:
            with open(stat_dict_path, 'w', encoding='utf8') as f:
                json.dump(stat_dict, f)
        if stat_dict[key] >= batch_num:  # 判断数量是否达到批量阈值
            return 1, stat_dict
        else:
            return 0, stat_dict
    elif left_h >= h_serious_thr:  # 左峰高度达到高度严重阈值
        stat_dict[key] += 1
        stat_count = len(list(stat_dict.values()))
        if stat_count > 0 and stat_count % 10:
            with open(stat_dict_path, 'w', encoding='utf8') as f:
                json.dump(stat_dict, f)
        return 2, stat_dict
    else:
        return 0, stat_dict
