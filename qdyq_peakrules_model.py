#!/usr/bin/env python
# -*- coding: utf-8 -*-
''''''
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os
import json
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import statistics

from utils.data_definition import defect_list
from utils.ob_utils import data_preprocess, segment_r_derivative, search_all_peaks, plot
from utils.strategy_utils import strategy_1,strategy_1AEQ
from sputils.read_rui import get_rui_from_dictAEQ
from sputils.rui_params import get_weld_sections_v2
from sputils.math_utils import curve_diff, bsn_rui_least_square_smooth
from sputils.math_utils import find_firstderiv_sections_base_sec, check_num_sign
from sputils.datetime_utils import str_to_datetime, datetime_to_str
''''''
def sorted_0(item):
    return item[0]

def sorted_2(item):
    return item[2]
''''''
# 数据预处理进行的功能函数
def data_funcAEQ(data_q=None,data_model=None):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    data = data_q
    rui = get_rui_from_dictAEQ(data['data'],data_model=data_model)
    # 对rui曲线进行数据预处理
    is_weld2, r_sec_ori, r_sec, r_deriv_1, sp_sections_dict = data_preprocessAEQ(rui)
    if not is_weld2:
        pass
    data['rui'] = rui
    data['r_sec'] = r_sec
    data['r_sec_ori'] = r_sec_ori
    data['r_deriv_1'] = r_deriv_1
    data['sp_sections_dict'] = sp_sections_dict
    # 矩阵放入rui_q队列
    return data

''''''
# 策略1函数
def strategy_1_funcAEQ(stt_1_q,h_batch_thr,h_serious_thr,w_thr,batch_num,stat_dict_path):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    if not osp.isfile(stat_dict_path):
        stat_dict = {}
    else:
        _, extname = osp.splitext(stat_dict_path)
        if extname.upper() == "JSON":
            with open(stat_dict_path, 'r', encoding='utf8') as f:
                stat_dict = json.load(f)
        else:
            stat_dict = {}
    data = stt_1_q  # 获取数据
    ori_data = data['data']
    all_peaks = data['all_peaks']
    ob_code, stat_dict = strategy_1AEQ(
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
        result = 'OK'
    else:
        result = 'NOK'
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
        return data
    else:
        return None
''''''
# 画图
def plotAEQ(data, rui, all_peaks, sp_sections_dict, save_boot):
    ori_data = data['data']
    sp2_head_t = sp_sections_dict['weld2']['t'][0]
    try:
        plot_name = f'{ori_data["serial_no"]}#{ori_data["spot_tag"]}#{ori_data["id"]}'
        plot_name2 = f'{ori_data["serial_no"]}#{ori_data["spot_tag"]}#{ori_data["id"]}'
    except:
        plot_name = f'{ori_data["serial_no_"]}#{ori_data["spotTag_"]}#{ori_data["id"]}'
        plot_name2 = f'{ori_data["serial_no_"]}#{ori_data["spotTag_"]}#{ori_data["id"]}'
    datetimestr = ori_data['rui_time']
    dateobj = str_to_datetime(datetimestr)
    print_str = datetime_to_str(dateobj, format='%Y-%m-%d')
    save_dir = osp.join(save_boot, print_str)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    fig = Figure(figsize=(12, 7), dpi=60)
    ax = fig.subplots(3, 1)

    fig.suptitle(
        (f'{plot_name} \n'
         # f'RFID={ori_data["RFID_1"]} \n'
         f'Defect: {data["result"]}, {data["batch_serious"]}'),
        fontsize=15)
    fontsize = 20
    curve_name = "I"
    label_str = f"{curve_name}"
    ax[0].plot(rui[curve_name], color='b', label=label_str)
    ax[0].legend(prop={'size': fontsize})

    curve_name = "U"
    label_str = f"{curve_name}"
    ax[1].plot(rui[curve_name], color='b', label=label_str)
    ax[1].legend(prop={'size': fontsize})

    curve_name = "R"
    label_str = f"{curve_name}"
    ax[2].plot(rui[curve_name], color='b', label=label_str)
    for i_p, (peak_h, ob_obj) in enumerate(all_peaks):
        xmin, xmax = ob_obj['box'][0], ob_obj['box'][2]
        ymin, ymax = round(ob_obj['box'][1], 4), round(ob_obj['box'][3], 4)
        left_h = ob_obj['left_h']
        ax[2].vlines(x=sp2_head_t + xmin, ymin=ymin, ymax=ymax, colors='r', ls='-')
        ax[2].vlines(x=sp2_head_t + xmax, ymin=ymin, ymax=ymax, colors='r', ls='-')
        ax[2].axvline(x=sp2_head_t + ob_obj['max_t'], c='r', ls='--')
        ax[2].hlines(y=ymin, xmin=sp2_head_t + xmin, xmax=sp2_head_t + xmax, colors='r', ls='-')
        ax[2].hlines(y=ymax, xmin=sp2_head_t + xmin, xmax=sp2_head_t + xmax, colors='r', ls='-')
        if peak_h > 0.01:
            ax[2].text(sp2_head_t + xmin, ymax + 0.01, f'abnormal, {round(left_h, 4)}', c='r')
    ax[2].legend(prop={'size': fontsize})
    save_path = osp.join(save_dir, f'{plot_name2}.jpg')
    fig.savefig(save_path)

''''''
# RUI曲线预处理
def data_preprocessAEQ(rui):
    sp_sections_dict = get_weld_sections_v2(rui)
    is_weld2 = False
    r_sec = []
    r_sec_ori = []
    r_deriv_1 = []
    if 'weld2' in sp_sections_dict:
        is_weld2 = True
        # 曲线平滑预处理
        r_sec_ori = sp_sections_dict['weld2']['r']
        r_sec = bsn_rui_least_square_smooth(r_sec_ori)
        # 求导
        r_deriv_1, r_deriv_2 = curve_diff(r_sec)
    return is_weld2, r_sec_ori, r_sec, r_deriv_1, sp_sections_dict
''''''
# 对一阶导数曲线按斜率程度进行分段，每段设置斜率级别
def segment_r_derivative(r_sec, r_deriv_1):
    # 对一阶导数曲线按斜率程度进行分段，每段设置斜率级别
    r_deriv_1_groups, r_deriv_1_sections = find_firstderiv_sections_base_sec(r_deriv_1)
    deriv_sections = r_deriv_1_sections
    len_r_sec = len(r_sec)

    # 斜率级别1和2进行合并，统称为1。0表示平缓段，1表示正向倾斜段，-1表示反向倾斜段
    len_secs = len(deriv_sections)
    level_ls = []
    for ii in range(len_secs):
        item = deriv_sections[ii]
        this_sec, this_lev = item
        this_time = this_sec[1] - this_sec[0] + 1
        is_pre = False
        is_next = False
        if ii > 0:
            pre_item = deriv_sections[ii - 1]
            pre_sec, pre_lev = pre_item
            is_pre = True
        if ii < len_secs - 1:
            next_item = deriv_sections[ii + 1]
            next_sec, next_lev = next_item
            is_next = True
        if this_time < 3:
            if is_pre and not is_next and abs(pre_lev - this_lev) < 2:
                level_ls.append(pre_lev)
            elif not is_pre and is_next:
                level_ls.append(this_lev)
            else:
                pre_lev = level_ls[ii - 1]
                if pre_lev == next_lev:
                    level_ls.append(pre_lev)
                else:
                    level_ls.append(this_lev)
        else:
            level_ls.append(this_lev)

    for ii in range(len_secs):
        deriv_sections[ii][1] = level_ls[ii]
    # 斜率同级别合并。多段连续的1合并为一段更长的1，0段也做相同处理。如果两个1中间夹杂了很短的0，则忽略0，按连续的1段进行合并。
    new_deriv_secs = []
    tmp = []
    last_t = 0
    if len_secs > 1:
        for ii in range(len_secs - 1):  # 重新标记斜率等级
            if len(tmp) == 0:
                f_sec, f_lev = deriv_sections[ii]  # 当前区，第1个区
            else:
                f_sec, f_lev = tmp[0]  # 当前区，第1个区
            s_sec, s_lev = deriv_sections[ii + 1]

            if check_num_sign(f_lev) == check_num_sign(s_lev) or (
                    check_num_sign(f_lev) != check_num_sign(s_lev) and s_sec[0] == s_sec[1]):
                if len(tmp) == 0:  # 如果tmp为空，则当前区与下一个区合并
                    tmp.append([[f_sec[0], s_sec[1]], check_num_sign(f_lev)])
                else:  # 如果tmp不为空，则tmp内的区与下一个区合并
                    tmp[0][0][1] = s_sec[1]  # 只修改tmp的截至位置即可
            else:  # 不满足合并条件，则向新队列中加入区
                if len(tmp) > 0:  # 如果tmp中有区，则tmp区加入新队列
                    if tmp[0][0][1] - tmp[0][0][0] > 0:
                        new_deriv_secs.append(tmp[0])
                        last_t = new_deriv_secs[-1][0][1]
                    tmp.clear()  # 清空tmp
                else:  # tmp为空，则将当前区加入到新队列
                    if f_sec[1] > f_sec[0]:
                        new_deriv_secs.append([f_sec, check_num_sign(f_lev)])
                        last_t = new_deriv_secs[-1][0][1]
        if len(tmp) > 0:  # 循环结束，如果tmp不为空，则tmp加入至新队列
            if tmp[0][0][1] > tmp[0][0][0]:
                new_deriv_secs.append(tmp[0])
                last_t = new_deriv_secs[-1][0][1]
            tmp.clear()
        if last_t < s_sec[1]:  # 如果最后一个区未加入至新队列，则加入
            if s_sec[1] > s_sec[0]:
                new_deriv_secs.append([s_sec, check_num_sign(s_lev)])
    else:
        new_deriv_secs = deriv_sections
    # 提取各段的核心数据
    deriv_r_secs = []
    for i_sec, sec in enumerate(new_deriv_secs):
        head_t = sec[0][0]
        tail_t = sec[0][1]
        diff_cate = sec[1]
        head_r = float(r_sec[head_t])
        tail_r = float(r_sec[tail_t])
        slope_r = (tail_r - head_r) / (tail_t - head_t + 1)
        peaks_r = r_sec[head_t:tail_t + 1]
        min_r = float(np.min(peaks_r))
        max_r = float(np.max(peaks_r))
        max_loc_arr = np.where(peaks_r == max_r)[0]
        max_loc = int(max_loc_arr[0])

        max_t = max_loc + head_t
        # 段sec的序号，斜率类别，起始时间，结束时间，时间长度，起始电阻，结束电阻，电阻变化值，电阻平均变化率
        deriv_r_dict = {
            'i_sec': i_sec,
            'diff_cate': diff_cate,
            'head_t': head_t,
            'tail_t': tail_t,
            'sec_t': tail_t - head_t + 1,
            'time': [head_t, tail_t],
            'max_t': max_t,
            'head_r': head_r,
            'tail_r': tail_r,
            'min_r': min_r,
            'max_r': max_r,
            'box': [head_t, min_r, tail_t, max_r],
            'sec_r': tail_r - head_r,
            'slope_r': slope_r,
        }
        deriv_r_secs.append(deriv_r_dict)
    return deriv_r_secs
''''''
# 搜索所有的山
def search_hills(peaks):
    hills = []  # 保存山峰群中的每个山
    check_add_hill = []
    tmp_hill = []
    for this_ii, this_peak in enumerate(peaks):
        is_down = False  # 是否出现下坡
        # 已被验证过，跳过
        if this_peak['i_sec'] in check_add_hill:
            continue
        # 起始不为上坡，跳过
        if this_peak['diff_cate'] < 0:
            continue
        # 符合一个山的起始条件
        tmp_hill.append(this_peak)
        check_add_hill.append(this_peak['i_sec'])
        # 当前是最后一个段，直接加入hills
        if this_ii == len(peaks) - 1:
            hills.append(tmp_hill.copy())
            tmp_hill.clear()
            continue
        # 遍历后面的各段，找到属于this_ii这座山的段
        for next_ii in range(this_ii + 1, len(peaks)):
            next_peak = peaks[next_ii]

            # 正常的上坡、平缓、下坡
            if next_peak['diff_cate'] > 0:  # 上坡
                if is_down:  # 已经出现下坡，则之前的部分是一个山
                    hills.append(tmp_hill.copy())
                    tmp_hill.clear()
                    break
                else:  # 没有出现下坡，则是继续上坡
                    tmp_hill.append(next_peak)
                    check_add_hill.append(next_peak['i_sec'])
            elif next_peak['diff_cate'] == 0:  # 平缓，直接添加为山的段
                tmp_hill.append(next_peak)
                check_add_hill.append(next_peak['i_sec'])
            else:  # 如果是下坡
                is_down = True
                tmp_hill.append(next_peak)
                check_add_hill.append(next_peak['i_sec'])
    # 处理残留在tmp_hill的段，为一个山·
    if len(tmp_hill) > 0:
        hills.append(tmp_hill.copy())
        tmp_hill.clear()
    # 对小山进行优化,去除尾部的平坦部分
    for i_h in range(len(hills)):
        is_up = False
        is_down = False
        hill = hills[i_h]
        for i_s in range(len(hill)):
            sec = hill[i_s]
            if sec['diff_cate'] > 0:
                is_up = True
            if sec['diff_cate'] < 0:
                is_down = True
        if hill[-1]['diff_cate'] == 0 and is_down:
            hill.pop(-1)
        if hill[-1]['diff_cate'] == 0 and not is_down and hill[-1]['time'][1] - hill[-1]['time'][0] > 50:
            hills.pop(i_h)
    return hills
''''''
# 搜索一个山峰
def search_peak(hill,r_sec_ori,r_sec,edge_float_range,):
    '''
    搜索每座小山的山峰。输入的电阻段包括原始值和平滑后的值。原始值能够更精准的计算最值和检测框高度。平滑值的求导值对判断上坡下坡更有利。
    :param hill:
    :param r_sec_ori: 原始电阻段，用于计算主要统计值
    :param r_sec: 平滑后的电阻段，用于求导数
    :param edge_float_range:
    :return:
    '''
    whead_t = hill[0]['head_t']  # 整个山峰段的起始时间
    wtail_t = hill[-1]['tail_t']  # 整个山峰段的结束时间
    time_sec = [whead_t, wtail_t]

    # 整理r曲线和计算重要数值
    peaks_r = []  # 山峰群的电阻
    peaks_r_for_diff = []  #
    for ob_dict in hill:
        head_t = ob_dict['head_t']
        tail_t = ob_dict['tail_t']
        peaks_r.extend(r_sec_ori[head_t:tail_t + 1])  # 山峰电阻列表，保存电阻原始值
        peaks_r_for_diff.extend(r_sec[head_t:tail_t + 1])  # 用于求导的山峰电阻列表，保存平滑后的电阻值

    peaks_r = np.array(peaks_r)
    min_r = np.min(peaks_r)
    max_r = np.max(peaks_r)
    max_loc_arr = np.where(peaks_r == max_r)[0]
    max_loc = max_loc_arr[0]
    head_r = peaks_r[0]
    tail_r = peaks_r[-1]
    max_t = max_loc + whead_t

    # 找小山
    max_edge_r = np.max([head_r, tail_r])  # 最大边缘值
    tmp_float_range = edge_float_range
    small_diff = np.diff(peaks_r_for_diff)  # 电阻局部区域的导数曲线
    abs_big_head = whead_t
    abs_big_tail = wtail_t
    hills = []
    is_extend = True  # 是否扩展浮动值收紧小山范围
    # 寻找最大山峰的头和尾，用山脚值+浮动范围筛选出多个候选值，经过进一步筛选得到大山的候选值
    # 通过循环，扩大浮动范围值，用来扩大山脚边界范围
    for _ in range(10):
        # 用max_edge_r和浮动范围值筛选相对坐标点
        screen_field = \
            np.where((peaks_r < max_edge_r + tmp_float_range) & (peaks_r > max_edge_r - tmp_float_range))
        # 如果筛选失败，继续扩大浮动范围值
        if screen_field[0].shape[0] == 0:
            continue
        # 如果最后一位坐标刚好是山峰的结尾，则需要重新赋值，防止用diff结果对原坐标进行定位时少一位
        if screen_field[0][-1] == peaks_r.shape[0] - 1:
            screen_field[0][-1] = peaks_r.shape[0] - 2
        # 筛选坐标转为list
        screen_field = screen_field[0].tolist()
        # 保存各个小山可能的起点或者终点，相临两点存为一组
        psb_foots = []  # 保存可能是小山的起点和终点
        tmp_field = []  # 临时保存
        for this_i in range(len(screen_field)):
            this_v = screen_field[this_i]
            if this_i == len(screen_field) - 1:
                tmp_field.append(this_v)
                psb_foots.append(tmp_field.copy())
                tmp_field.clear()
                break
            next_i = this_i + 1
            next_v = screen_field[next_i]
            if next_v - this_v > 1:
                tmp_field.append(this_v)
                psb_foots.append(tmp_field.copy())
                tmp_field.clear()
            else:
                tmp_field.append(this_v)
        # 小山
        # 保存所有的小山峰
        hill_head = -1
        hill_tail = -1
        for foot_list in psb_foots:
            screen_diff = small_diff[foot_list]
            upslope = np.where(screen_diff > 0)
            downslope = np.where(screen_diff < 0)
            # 为每段脚列表分配类型
            foot_ls_type = -1  # 脚列表类型
            if upslope[0].shape[0] > 0 and downslope[0].shape[0] == 0:  # 上坡
                foot_ls_type = 1
            elif upslope[0].shape[0] == 0 and downslope[0].shape[0] > 0:  # 下坡
                foot_ls_type = 2
            elif upslope[0].shape[0] > 0 and downslope[0].shape[0] > 0:  # 即存在上坡又存在下坡
                foot_ls_type = 3
            # 根据脚类型形成小山
            if foot_ls_type == 1:  # 上坡
                hill_head = foot_list[upslope[0][-1]]
            elif foot_ls_type == 2:  # 下坡
                hill_tail = foot_list[downslope[0][0]]
            elif foot_ls_type == 3:  # 上下坡都存在
                if hill_head >= 0:  # 已记录山的开头
                    hill_tail = foot_list[downslope[0][0]]
                elif hill_head < 0:  # 未记录山的开头
                    hill_head = foot_list[upslope[0][-1]]

            # 查找小山
            if hill_head >= 0 and hill_tail >= 0 and hill_head < hill_tail:
                hill_top = np.max(peaks_r[hill_head:hill_tail])
                hill_foot = np.min(peaks_r[hill_head:hill_tail])
                hill_hight = hill_top - hill_foot
                hills.append([hill_head, hill_tail, hill_hight])
                hill_head = -1
                hill_tail = -1
        if len(hills) > 0:  # [hill_head, hill_tail, hill_hight]
            hills.sort(key=sorted_2, reverse=True)
            big_head, big_tail = hills[0][0:2]
            if big_tail - big_head < 50 or not is_extend:
                break
            elif is_extend and 50 <= big_tail - big_head <= 150:
                tmp_float_range += 0.01
                is_extend = False
                hills.clear()
            elif is_extend and big_tail - big_head > 150:
                tmp_float_range += 0.015
                is_extend = False
                hills.clear()
        tmp_float_range += 0.001
    if len(hills) > 0:
        hills.sort(key=sorted_2, reverse=True)
        big_head, big_tail = hills[0][0:2]
        abs_big_head = big_head + time_sec[0]
        abs_big_tail = big_tail + time_sec[0]

    # xmin, xmax = abs_big_head, abs_big_tail
    # ymin, ymax = max_edge_r, max_r
    xmin = whead_t  # xmin是hill的起点，而不是abs_big_head
    xmax = abs_big_tail  # xmax是peak尾部abs_big_tail
    box_r = r_sec_ori[abs_big_head:abs_big_tail + 1]
    # ymin = r_sec_ori[abs_big_head] if r_sec_ori[abs_big_head] > r_sec_ori[abs_big_tail] else r_sec_ori[abs_big_tail]
    ymin = min([r_sec_ori[abs_big_head], r_sec_ori[abs_big_tail], min_r])
    ymax = np.max(box_r)
    ob_obj = {
        'time': time_sec,
        'max_t': max_t,
        'head_r': head_r,
        'tail_r': tail_r,
        'min_r': min_r,
        'max_r': max_r,
        'upslope': (max_r - head_r) / (max_t - time_sec[0]),
        'downslope': (tail_r - max_r) / (time_sec[1] - max_t + 1.0e-5),
        'peak_range': [abs_big_head, abs_big_tail],
        'peak_foot': max_edge_r,
        'box': [xmin, ymin, xmax, ymax],
        'left_h': ymax - r_sec_ori[whead_t],
        'right_h': ymax - r_sec_ori[wtail_t],
    }
    return ob_obj
''''''
# 搜索所有的山峰
def search_all_peaks(r_sec_ori,r_sec,deriv_r_secs,first_door=40,second_door=100,start_min_len=5,flat_max_len=170,flat_func_len=60,
        downhill_min_len=8,edge_float_range=0.001,early_ratio=4.0e-4,):
    # 保存所有的山峰
    all_peaks = []
    # 搜索异常山脉，山脉由很多山组成，不同山脉的边界是长的平坦区域
    check_add_mountain = []  # 保存已验证过的山，用于判断
    # 遍历各坡段
    for this_i in range(len(deriv_r_secs) - 1):
        this_item = deriv_r_secs[this_i]
        is_mountain = False  # 是否为群山
        mountain = []  # 保存山脉，山脉由很多山组成，不同山脉的边界是长的平坦区域
        is_up = False
        is_down = False
        # 寻找以this_i为起点段的山峰或群山
        # 起点段的特征是没有被验证过的、足够长的且不能太长的上升段。时间范围是[5, 25]
        # 结束段的特征是足够长的平坦段
        this_i_sec = this_item['i_sec']
        this_diff_cate = this_item['diff_cate']
        this_sec_t = this_item['sec_t']
        # 上坡的起始点要达到或超过first_door，防止早期曲线正常的起伏被误判为异常山峰。
        if this_item['head_t'] < first_door:
            continue
        # 该段如果已经验证过，则跳过。
        if this_i_sec in check_add_mountain:
            continue
        # 该段如果不是正斜率的上升坡，跳过
        if this_diff_cate <= 0:
            continue
        # 该段的上坡持续时间如果小于最小时间，跳过
        if this_sec_t < start_min_len:
            continue

        # 搜索以this_i为起点的山脉
        mountain.append(this_item)  # 满足条件，加入山峰队列
        check_add_mountain.append(this_item['i_sec'])  # 记录已验证段，加入验证队列
        # 遍历后面的段，看是否为该山峰的段
        for next_i in range(this_i + 1, len(deriv_r_secs)):
            next_item = deriv_r_secs[next_i]
            # 跳过已验证的段
            if next_item['i_sec'] in check_add_mountain:
                continue

            # 判断next_i是否为this_i山脉所属的段
            check_add_mountain.append(next_item['i_sec'])  # 记录已验证段，加入验证队列
            # 平缓段的判断，不符合山峰条件或者满足退出条件就退出该山峰的搜索
            if next_item['diff_cate'] == 0:  # 平缓段进行判断
                if next_item['sec_t'] > flat_max_len:  # 遇到了足够平坦的段，结束山脉this_i的寻找
                    break
                else:
                    if next_item['i_sec'] == len(deriv_r_secs) - 1:  # 平坦段是最后一个段，结束山脉寻找
                        break
                    else:
                        last_item = deriv_r_secs[next_i - 1]  # next_i的上一个段
                        next_next_item = deriv_r_secs[next_i + 1]  # next_i的下一个段
                        last_cate = last_item['diff_cate']  # 上一个段的类别
                        next_next_cate = next_next_item['diff_cate']  # 下一个段的类别
                        # 如果是上坡过程或者下坡过程，平缓段不易过长，阈值用flat_func_len。flat_func_len小于flat_max_len
                        if last_cate * next_next_cate > 0 and next_item['sec_t'] > flat_func_len:  # 均为1或均为-1
                            break
            # 最后一个段的判断
            if next_i == len(deriv_r_secs) - 1:  # 如果到了最后一个段，则结束
                # 最后段是平缓段，则不要该段
                if next_item['diff_cate'] == 0:
                    break
                mountain.append(next_item)
                is_mountain = True  # 整个this_i的大段的趋势无论是上升下降还是仅为上升，只要进入结尾段的判断，则整个段即为山峰
                break
            mountain.append(next_item)  # 在经过了各个条件的筛选后，next_i段为山峰this_i的一个段
            # 如果this_i的大段内存在下坡段且下坡段的长度足够长，则认为该this_i是山脉
            # 防止出现仅有上坡段没有下坡段这种情况。
            if next_item['diff_cate'] < 0 and next_item['sec_t'] >= downhill_min_len:
                is_mountain = True

        if len(mountain) == 1:
            if abs(mountain[0]['tail_t'] - len(r_sec_ori)) > 50:
                continue

        # 将一个大的山脉进行整理，找到多个山峰
        if (len(mountain) > 0 and is_mountain) or len(mountain) == 1:  # 存在下坡才能形成山峰
            # 在群山中找到各个山
            hills = search_hills(mountain)
            # 找到每个山的山峰
            for hill in hills:
                peak = search_peak(
                    hill,
                    r_sec_ori,
                    r_sec,
                    edge_float_range,
                )
                all_peaks.append(peak)
    # 对所有山峰进行过滤
    # 对于发生时间比较早的山峰，要通过更严格的斜率进行过滤，防止出现前期电阻的正常快速上升产生的误判
    for i_peak in range(len(all_peaks) - 1, -1, -1):
        peak = all_peaks[i_peak]
        bbox = peak['box']
        ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
        # 山峰移除条件：
        # （1）如果山峰起始时间小于second_door
        # 去除该条件：（2）在（1）的基础上山峰的上坡斜率低于slope_thr
        # （2）在（1）的基础上小山的高宽比低于early_ratio
        if peak['time'][0] <= second_door:
            if ratio < early_ratio:
                all_peaks.remove(peak)
    result = [(peak['left_h'], peak) for peak in all_peaks]
    result.sort(key=sorted_0, reverse=True)

    return result
''''''
# 过烧计算进程的核心函数
def ob_func(ob_q=None):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    data = ob_q  # 获取数据
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
    return data
''''''
# 画图的核心函数
def plot_func(plot_q, save_dir):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    data = plot_q # 获取数据
    rui = data['rui']
    all_peaks = data['all_peaks']
    sp_sections_dict = data['sp_sections_dict']
    plot(data, rui, all_peaks, sp_sections_dict, save_dir)

# 画图的核心函数
def plot_funcAEQ(plot_q, save_dir):
    # 在各队列间，数据之间的传输基本格式：{'id':xxx, 'data':xxx, 'result':xxx, other_key:xxx}
    # 创建logger实例
    data = plot_q # 获取数据
    rui = data['rui']
    all_peaks = data['all_peaks']
    sp_sections_dict = data['sp_sections_dict']
    plotAEQ(data, rui, all_peaks, sp_sections_dict, save_dir)
''''''
def peakrules_model(data_ins:dict,on_off=None,data_model=None,save_flag=None,save_file=None,save_file_dict=None):
    #data_ins为输入数据，必须是字典
    #on_off为模型开关，True为开，False为关
    #data_model，数据来源，如果是'file',则代表从文件里取的数据，ui曲线按字符处理，如果是'ck',代表从ck取数据，ui曲线按列表处理
    #save_flag=True,则为保存异常图片，save_file为保存异常图片的文件夹、比如r'D:\Git_prj\异常尖峰识别规则_青岛全厂\AB_20230615_save_p'，
    #save_file_dict必须提供，比如r'D:\Git_prj\异常尖峰识别规则_青岛全厂\AB_20230615_save_p\stat_dict.json',其中stat_dict.json名字不要变
    query_spot_id = data_ins['id']
    query_spot_data = data_ins
    data_in = {}
    if on_off==True:
        try:  # 判定曲线是否异常，并生成对应图片
            data_in.update({'id': query_spot_id, 'data': query_spot_data})
            data_out = data_funcAEQ(data_q=data_in,data_model=data_model)
            stt_1_q = ob_func(ob_q=data_out)
            plot_q = strategy_1_funcAEQ(stt_1_q=stt_1_q,
                                        h_batch_thr=0.04,
                                        h_serious_thr=0.065,
                                        w_thr=30,
                                        batch_num=3,
                                        stat_dict_path=save_file_dict)
            if plot_q != None and save_flag==True:
                plot_funcAEQ(plot_q=plot_q,save_dir=save_file)
            else:
                pass

            if plot_q !=None:
                return {'result':'nok'}
            else:
                return {'result':'ok'}
        except:
            return {'result':'数据异常'}
    else:
        return {'result':None, 'result_type': '合格'}#当模型关闭的时候，配合其他模型的整体数据结果，需要用这个做判断，不整合该模型结果
''''''
if __name__=='__main__':
    file_=r"D:\Git_prj\异常尖峰识别规则_青岛全厂\test_data\CK_20251015.csv"
    try:
        file_data = pd.read_csv(file_,encoding='gbk')
    except:
        file_data = pd.read_excel(file_,sheet_name='Sheet1')
    data_gb=file_data.groupby('spot_tag')
    for ks,v in data_gb:
        print(ks)
        # 得到当前一条数据曲线
        for k in range(len(v)):#遍历某个文件所有数据
            #数据预处理
            query_spot_data = v.iloc(0)[k].to_dict()
            result=peakrules_model(data_ins=query_spot_data,on_off=True,data_model='file',save_flag=True,
                            save_file=r'D:\Git_prj\异常尖峰识别规则_青岛全厂\test_data_save_p',
                            save_file_dict=r'D:\Git_prj\异常尖峰识别规则_青岛全厂\test_data_save_p\stat_dict.json')
            print(result)