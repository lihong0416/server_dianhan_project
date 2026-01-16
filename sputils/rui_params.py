#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2022/6/12 15:05

import math
import inspect
import numpy as np
from collections import Counter
from .math_utils import find_coord_continuous_section, curve_diff, find_firstderiv_sections_base_sec
from .math_utils import curve_diff_1, check_num_sign, bsn_rui_least_square_smooth
from .phy_utils import get_rui_sec_actual_parmas, get_rui_simple_feats_3, get_rui_simple_feats_2_2


def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for var_name, var_val in callers_local_vars:
        if var_val is variable:
            return var_name
    return False

def create_default_sections(curve_length):
    """Create default section structure when curve analysis fails"""
    return {
        'rui': {'I': [0]*curve_length, 'U': [0]*curve_length, 'R': [0]*curve_length, 'P': [0]*curve_length},
        'time': [(1, (0, curve_length-1))],
        'pulse_count': 0,
        'weld1': {},
        'weld2': {
            't': (0, curve_length-1),
            'i': [0]*curve_length,
            'u': [0]*curve_length, 
            'r': [0]*curve_length,
            'p': [0]*curve_length
        },
        'weld3': {},
        'weld_count': 1,
    }

def get_weld_sections_v2(rui):
    I = rui['I']
    U = rui['U']
    R = rui['R']
    P = rui['P']
    # sI = rui['I'][0:rui_len] # 长度受限
    # sR = rui['R'][0:rui_len] # 长度受限

    # ADD: Comprehensive validation
    try:
        # Validate all curves exist and have data
        for curve_name, curve_data in [('I', I), ('U', U), ('R', R), ('P', P)]:
            if not curve_data or len(curve_data) < 50:
                print(f"Warning: Invalid {curve_name} curve data")
                return create_default_sections(len(R) if R else 1000)
        
        # Validate resistance data specifically
        R_array = np.array(R, dtype=float)
        if np.all(R_array <= 0) or np.std(R_array) < 0.0001:
            print("Warning: Invalid resistance pattern detected")
            return create_default_sections(len(R))
            
    except Exception as e:
        print(f"Error validating curve data: {e}")
        return create_default_sections(1000)

    zero_sections = []  # 存储非通电的时间段
    energy_sections = []  # 存储通电的时间段
    is_pulse = False  # 判断是否为脉冲焊
    # 获得电阻为0的区域
    zero_list = np.where(np.array(R) < 0.001)[0].tolist()

    if not zero_list:
        print("Warning: No zero resistance periods found, treating as continuous welding")
        # If no zero periods, treat entire curve as one welding section
        energy_sections.append((0, len(R) - 1))
        
        # Skip to section creation logic
        sp_sections_dict = {
            'rui': rui,
            'time': [(1, (0, len(R)-1))],
            'pulse_count': 0,
            'weld1': {},
            'weld2': {
                't': (0, len(R)-1),
                'i': I,
                'u': U,
                'r': R,
                'p': P
            },
            'weld3': {},
            'weld_count': 1,
        }
        return sp_sections_dict

    head = zero_list[0]
    end = -1
    for i_this in range(len(zero_list) - 1):
        i_next = i_this + 1
        v_this = zero_list[i_this]
        v_next = zero_list[i_next]
        if v_next - v_this > 1:
            end = v_this
            if v_this > head:
                zero_sections.append((head, end))
            head = v_next
            end = -1
    if head > -1 and end == -1 and head < zero_list[len(zero_list) - 1]:
        zero_sections.append((head, zero_list[len(zero_list) - 1]))
    # 根据电阻非0的条件获得疑似通电段
    suspected_energy_sections = []
    valid_head = 0  # 有效焊接区的头
    if len(zero_sections) == 0:
        energy_sections.append((0, len(R) - 1))
    else:
        for zero_sec in zero_sections:
            if zero_sec[1] - zero_sec[0] > 1 and zero_sec[0] - valid_head > 1:
                sec = (valid_head, zero_sec[0] - 1)
                suspected_energy_sections.append(sec)
                valid_head = zero_sec[1] + 1
        if valid_head < len(R) - 1:
            suspected_energy_sections.append((valid_head, len(R) - 1))
        # 筛除疑似通电段中的伪通电段

        for i_sec in range(len(suspected_energy_sections)):
            sec = suspected_energy_sections[i_sec]
            # 筛选条件1：各段长度大于100，因为伪通电段比较短，会错筛掉预焊
            # 筛选条件2：段的起始时间小于900。因为伪通电段都在1000左右开始才出现
            # 脉冲焊的某段可能出现在900之后，但是正常的脉冲焊时间会长于99
            # 结合1和2，小于900的短时间焊接如预焊会被保留。万一有起始时间高于900的脉冲焊段也能被保留。
            # 未筛除掉的是长时间的伪通电段。后面会根据它是否为脉冲焊对它进行筛除。
            if sec[1] - sec[0] > 99 or sec[0] < 900:
                energy_sections.append(sec)
        # 进一步筛除伪通电段
        # 统计各段通电时间长度，用模糊的方式统计，即长度相差低于15的就认为是时间长度相同
        len_counter_dict = {}
        for i_sec, sec in enumerate(energy_sections):
            len_sec = sec[1] - sec[0] + 1
            is_update = False
            for key in len_counter_dict:
                if abs(len_sec - key) < 15:
                    len_counter_dict[key]['count'] += 1
                    if i_sec - len_counter_dict[key]['last'] == 1:
                        len_counter_dict[key]['continuous'] = True
                    len_counter_dict[key]['last'] = i_sec
                    is_update = True
            if not is_update:
                len_counter_dict[len_sec] = {}
                len_counter_dict[len_sec]['count'] = 1  # 此长度的数量
                len_counter_dict[len_sec]['last'] = i_sec  # 此长度上一段的id
                len_counter_dict[len_sec]['continuous'] = False  # 此长度当前段与上一个是否在id上是连续的

        # 判断是否为脉冲
        for this_len in len_counter_dict:
            count = len_counter_dict[this_len]['count']
            is_continuous = len_counter_dict[this_len]['continuous']
            if count > 2 and is_continuous:
                is_pulse = True
                pulse_len = this_len

        for i_ in range(len(energy_sections)):
            i_sec = len(energy_sections) - i_ - 1
            sec = energy_sections[i_sec]
            len_sec = sec[1] - sec[0] + 1
            if sec[0] >= 900:  # 段的起始时间大于等于900
                if is_pulse:  # 如果是脉冲，看它是否为脉冲段
                    if abs(len_sec - pulse_len) > 15:  # 根据段长与脉冲段的长度比较判断该段是否为脉冲
                        energy_sections.remove(sec)
                else:  # 如果不是脉冲，直接删除该段
                    energy_sections.remove(sec)
    # 点焊分段结果
    sp_sections_dict = {
        'rui':rui,
        'time':[],
        'pulse_count': 0,
        'weld1': {},
        'weld2': {},
        'weld3': {},
        'weld_count': 0,
    }
    # 把时间信息存到分段结果中
    head = 0
    for e_sec in energy_sections:
        if e_sec[0] !=head:
            zero_sec=(0,(head,e_sec[0]-1))
            energy_sec=(1,e_sec)
            sp_sections_dict['time'].append(zero_sec)
            sp_sections_dict['time'].append(energy_sec)
        else:
            energy_sec = (1, e_sec)
            sp_sections_dict['time'].append(energy_sec)
        head = e_sec[1] + 1
    if head < len(R):
        zero_sec = (0, (head, len(R)-1))
        sp_sections_dict['time'].append(zero_sec)

    # 判断第一个通电段是只属于一次焊接还是两次焊接，防止预焊与主焊之间冷却时间为0的情况
    # 判断第一段，第一段最复杂
    sec_0 = energy_sections[0]
    len_0 = sec_0[1] - sec_0[0]
    if sec_0[1] - sec_0[0] <= 70:  # 第一段焊接时间小于70ms，则认为是预焊（第一次焊接）
        sp_sections_dict['weld1']['t'] = sec_0
        sp_sections_dict['weld1']['i'] = I[sec_0[0]:sec_0[1] + 1]
        sp_sections_dict['weld1']['u'] = U[sec_0[0]:sec_0[1] + 1]
        sp_sections_dict['weld1']['r'] = R[sec_0[0]:sec_0[1] + 1]
        sp_sections_dict['weld1']['p'] = P[sec_0[0]:sec_0[1] + 1]
        sp_sections_dict['weld_count'] += 1
    elif sec_0[1] - sec_0[0] > 70:  # 仅在第一段长度大于70情况下进行判断
        sec_i = I[sec_0[0] + 20:sec_0[1] - 20]  # 该段前20ms和后20ms都不要了
        diff = np.diff(sec_i)
        max_diff = np.max(np.abs(diff))
        if max_diff > 0.5:  # 有分割点，前半段是预焊，后半段是主焊
            split_point = np.argwhere(np.abs(diff) == max_diff)[0][0]
            sp_sections_dict['weld1']['t'] = (sec_0[0], split_point - 1)
            sp_sections_dict['weld1']['i'] = I[sec_0[0]:split_point]
            sp_sections_dict['weld1']['u'] = U[sec_0[0]:split_point]
            sp_sections_dict['weld1']['r'] = R[sec_0[0]:split_point]
            sp_sections_dict['weld1']['p'] = P[sec_0[0]:split_point]
            sp_sections_dict['weld_count'] += 1
            sp_sections_dict['weld2']['t'] = (split_point, sec_0[1])
            sp_sections_dict['weld2']['i'] = I[split_point:sec_0[1] + 1]
            sp_sections_dict['weld2']['u'] = U[split_point:sec_0[1] + 1]
            sp_sections_dict['weld2']['r'] = R[split_point:sec_0[1] + 1]
            sp_sections_dict['weld2']['p'] = P[split_point:sec_0[1] + 1]
            sp_sections_dict['weld_count'] += 1
        else:  # 没有分割点
            if is_pulse:  # 如果是脉冲，需要跟时间判断第一段是否为脉冲
                len_1 = energy_sections[1][1] - energy_sections[1][0]
                if abs(len_1 - len_0) < 15:
                    sp_sections_dict['weld2']['t'] = sec_0
                    sp_sections_dict['weld2']['i'] = I[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['u'] = U[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['r'] = R[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['p'] = P[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld_count'] += 1
                else:  # 如果没有脉冲，则第一段是预焊
                    sp_sections_dict['weld1']['t'] = sec_0
                    sp_sections_dict['weld1']['i'] = I[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['u'] = U[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['r'] = R[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['p'] = P[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld_count'] += 1
            else:  # 如果不是脉冲，时间超过160就算为主焊，低于160算为预焊
                if len_0 < 160:
                    sp_sections_dict['weld1']['t'] = sec_0
                    sp_sections_dict['weld1']['i'] = I[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['u'] = U[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['r'] = R[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld1']['p'] = P[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld_count'] += 1
                else:
                    sp_sections_dict['weld2']['t'] = sec_0
                    sp_sections_dict['weld2']['i'] = I[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['u'] = U[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['r'] = R[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld2']['p'] = P[sec_0[0]:sec_0[1] + 1]
                    sp_sections_dict['weld_count'] += 1

    # 第二段至后面的段
    if len(energy_sections) > 1:
        for i_sec in range(1, len(energy_sections)):
            sec = energy_sections[i_sec]  # 取出时间段
            # 如果焊接段的编号被占用，则用下一位
            for i_weld in range(2, 10):
                keyname = f"weld{i_weld}"
                if keyname in sp_sections_dict and len(sp_sections_dict[keyname]) > 0:
                    continue
                if keyname not in sp_sections_dict:
                    sp_sections_dict[keyname] = {}
                sp_sections_dict[keyname]['t'] = sec
                sp_sections_dict[keyname]['i'] = I[sec[0]:sec[1] + 1]
                sp_sections_dict[keyname]['u'] = U[sec[0]:sec[1] + 1]
                sp_sections_dict[keyname]['r'] = R[sec[0]:sec[1] + 1]
                sp_sections_dict[keyname]['p'] = P[sec[0]:sec[1] + 1]
                sp_sections_dict['weld_count'] += 1
                break
    # 设置脉冲值
    if is_pulse:
        if len(sp_sections_dict['weld1']) > 0:
            sp_sections_dict['pulse_count'] = sp_sections_dict['weld_count'] - 1
        else:
            sp_sections_dict['pulse_count'] = sp_sections_dict['weld_count'] - 1
    return sp_sections_dict


def get_weld_sections(rui, outlier_thr=0.5, weldstage_1_t_thr=160, is_relative=False):
    '''
    将焊接曲线分为预焊（焊接1）、主焊1（焊接2）、主焊2（焊接3）。
    提取冷却时间。
    '''
    # 用于保存各段分隔结果的字典
    total_weld_dict = {}  # 整个焊接过程
    weldstage_1_dict = {}  # 焊接1，即预焊
    weldstage_2_dict = {}  # 焊接2，即主焊1
    weldstage_3_dict = {}  # 焊接3，即主焊2
    tail_sec_dict = {}  # 尾部区，不算焊接
    cool_1_dict = {'is': False}  # 冷却时间1，即预焊和主焊1之间的冷却时间
    cool_1_dict['name'] = get_variable_name(cool_1_dict)
    cool_1_dict['is_cur'] = False  # 是否存在>0的曲线段，冷却时间无电流电压，is_cur设为0
    cool_2_dict = {'is': False}  # 冷却时间2，即主焊1和主焊2之间的冷却时间
    cool_2_dict['name'] = get_variable_name(cool_2_dict)
    cool_2_dict['is_cur'] = False

    # 用于保存各焊接阶段的时间段
    total_weld_time_section = []
    weldstage_1_time_section = []
    weldstage_2_time_section = []
    weldstage_3_time_section = []
    tail_sec_time = []

    '''
    取电压曲线，电压曲线相比电流曲线，其0值更可信。
    '''
    u = rui['U']
    u = np.array(u)
    len_u = u.size
    '''
    寻找焊接过程中电压为0的区段，这是将焊接分区的主要间隔
    '''
    zero_loc = np.where(u == 0)
    zero_loc_ls = zero_loc[0].tolist()
    '''
    焊接过程中施加电压时间小于20ms，则认为没有进行焊接。
    '''
    if len(zero_loc_ls) > len_u - 20:
        return total_weld_dict, weldstage_1_dict, weldstage_2_dict, weldstage_3_dict, tail_sec_dict

    '''
    焊接过程存在电压，继续判断
    '''
    zero_groups, len_zero_groups = find_coord_continuous_section(zero_loc_ls)
    '''
    依据是否存在电压0段分别进行讨论
    '''
    check_tail = 985
    zero_time_in_weld = []
    weld_end = len_u - 1
    len_groups = len(zero_groups)
    for ii in range(len_groups):
        group = zero_groups[ii]
        len_g = len_zero_groups[ii]
        start = group[0]
        end = group[1]
        # 排除尾部情况
        if start < check_tail:
            # 起始时间必须大于0,排除焊接开始时电压为0的情况，该0时段不能作为3阶段焊接的依据
            # 0时段长度应大于7，避免焊接过程中短时间的电压异常0值造成的误分区
            if start > 0 and len_g > 7:
                zero_time_in_weld.append(group)
        else:
            weld_end = start
            break

    len_weld_zero = len(zero_time_in_weld)
    start_loc = 0
    if len_weld_zero > 0:  # 焊接过程存在电压为0的区段，这里称为0段
        '''
        焊接区与尾部区的分区。
        '''
        tail_sec_time = [zero_time_in_weld[len_weld_zero - 1][1], len_u - 1]
        total_weld_time_section = [0, zero_time_in_weld[len_weld_zero - 1][0] - 1]

        # 排除0区段在焊接记录时出现的问题
        if len_weld_zero > 0:
            if zero_time_in_weld[0][0] < 2:
                head_zero = zero_time_in_weld.pop(0)
                start_loc = head_zero[1]

        '''
        焊接分区
        焊接1区通常表示预焊，焊接2区通常表示主焊1，焊接3区主要表示主焊2或者回火
        根据0段将焊接区进行分隔。
        '''
        len_weld_zero = len(zero_time_in_weld)
        if len_weld_zero > 2:
            weldstage_1_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
            weldstage_2_time_section = [zero_time_in_weld[0][1] + 1, zero_time_in_weld[1][0] - 1]
            weldstage_3_time_section = [zero_time_in_weld[1][1] + 1, zero_time_in_weld[2][0] - 1]
        elif len_weld_zero == 2:
            if zero_time_in_weld[1][1] < check_tail:
                weldstage_1_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
                weldstage_2_time_section = [zero_time_in_weld[0][1] + 1, zero_time_in_weld[1][0] - 1]
                weldstage_3_time_section = [zero_time_in_weld[1][1] + 1, weld_end]
            else:
                if zero_time_in_weld[0][0] - start_loc < weldstage_1_t_thr:
                    weldstage_1_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
                    weldstage_2_time_section = [zero_time_in_weld[0][1] + 1, zero_time_in_weld[1][0] - 1]
                else:
                    weldstage_2_time_section = [start_loc, zero_time_in_weld[0][0]]
                    weldstage_3_time_section = [zero_time_in_weld[0][1] + 1, zero_time_in_weld[1][0] - 1]

        elif len_weld_zero == 1:
            if zero_time_in_weld[0][1] < check_tail:
                if zero_time_in_weld[0][0] - start_loc < weldstage_1_t_thr:
                    weldstage_1_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
                    weldstage_2_time_section = [zero_time_in_weld[0][1] + 1, weld_end]
                else:
                    weldstage_2_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
            else:
                if zero_time_in_weld[0][0] - start_loc < weldstage_1_t_thr:
                    weldstage_1_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
                else:
                    weldstage_2_time_section = [start_loc, zero_time_in_weld[0][0] - 1]
        elif len_weld_zero == 0:
            weldstage_2_time_section = [start_loc, weld_end]
    else:  # 焊接曲线不存在0段，认为整个过程均为焊接区
        total_weld_time_section = [0, len_u - 1]
        weldstage_2_time_section = [0, len_u - 1]

    '''
    使用电流判断焊接2区（主焊1）是否可分为预焊
    '''
    if (len(weldstage_1_time_section) == 0 and len(weldstage_2_time_section) > 0):
        I = rui['I']
        I = I.copy()

        tmp_i = I[weldstage_2_time_section[0]:weldstage_2_time_section[1] + 1]
        len_i = len(tmp_i)
        for i in range(30, len_i - 30, 1):  # 首尾30ms数据不要了，防止电流递增递减造成的误判
            x = len_i - 10 - i - 1
            y_r = tmp_i[x:x + 10]
            y_l = tmp_i[x - 10:x]
            mean_r = float(np.mean(y_r))
            mean_l = float(np.mean(y_l))
            if abs(mean_r - mean_l) > outlier_thr and start_loc < (x - 1):
                window = tmp_i[x - 10:x + 10]
                window = np.array(window)
                y_n = abs(window)  # 不进行归一化，仅取绝对值
                diff_1 = curve_diff_1(y_n)  # 求1阶导数
                abs_diff = abs(diff_1)  # 1阶导数绝对值
                max_diff = np.max(abs_diff)
                if max_diff > outlier_thr:
                    outliers = np.where(abs_diff == max_diff)
                    outliers_loc = outliers[0][0]
                    weldstage_1_time_section = [start_loc, x - 10 + outliers_loc - 1]
                    weldstage_2_time_section = [x - 10 + outliers_loc, weldstage_2_time_section[1]]
                    break
        # I = np.array(I)
        # if is_relative:
        #     y_n = I / np.max(abs(I))  # 如果使用相对值判断异常点，先取绝对值再进行归一化
        # else:
        #     y_n = abs(I)  # 不进行归一化，仅取绝对值
        # diff_1 = curve_diff_1(y_n)  # 求1阶导数
        # abs_diff = abs(diff_1)  # 1阶导数绝对值
        # outliers = np.where(abs_diff > outlier_thr)  # 根据1阶导数曲线判断异常点，该变量是数组
        # if len(outliers) > 0:
        #     outliers_ls = outliers[0].tolist()
        #     outliers_ls.sort()
        #     for outlier in outliers_ls:
        #         if outlier > 30:
        #             weldstage_1_time_section = [start_loc, outlier - 1]
        #             weldstage_2_time_section = [outlier, weldstage_2_time_section[1]]
        #             break

    '''
    初始化时间段键值对和判断是否存在的键值对
    '''
    # 处理一下可能的异常情况
    if len(weldstage_1_time_section) > 0 and weldstage_1_time_section[1] - weldstage_1_time_section[0] <= 0:
        weldstage_1_time_section = []
    if len(weldstage_2_time_section) > 0 and weldstage_2_time_section[1] - weldstage_2_time_section[0] <= 0:
        weldstage_2_time_section = []
    if len(weldstage_3_time_section) > 0 and weldstage_3_time_section[1] - weldstage_3_time_section[0] <= 0:
        weldstage_3_time_section = []
    total_weld_dict = {'time_section': total_weld_time_section,
                       'is': False,
                       'is_cur': True,
                       'name': get_variable_name(total_weld_dict)}
    weldstage_1_dict = {'time_section': weldstage_1_time_section,
                        'is': False,
                        'is_cur': True,
                        'name': get_variable_name(weldstage_1_dict)}
    weldstage_2_dict = {'time_section': weldstage_2_time_section,
                        'is': False,
                        'is_cur': True,
                        'name': get_variable_name(weldstage_2_dict)}
    weldstage_3_dict = {'time_section': weldstage_3_time_section,
                        'is': False,
                        'is_cur': True,
                        'name': get_variable_name(weldstage_3_dict)}

    tail_sec_dict = {'time_section': tail_sec_time,
                     'is': False,
                     'is_cur': True,
                     'name': get_variable_name(tail_sec_dict)}

    temp_ls = [total_weld_dict, weldstage_1_dict, weldstage_2_dict, weldstage_3_dict]
    for this_dict in temp_ls:
        if len(this_dict['time_section']) > 0:
            this_dict['is'] = True
            this_dict['time'] = this_dict['time_section'][1] - this_dict['time_section'][0] + 1
            for key in ['I', 'U', 'R', 'P']:
                this_dict[key] = rui[key][this_dict['time_section'][0]:this_dict['time_section'][1] + 1]

    for key in ['I', 'U', 'R', 'P']:
        if len(tail_sec_dict['time_section']) > 0:
            tail_sec_dict['is'] = True
            tail_sec_dict['time'] = tail_sec_dict['time_section'][1] - tail_sec_dict['time_section'][0] + 1
            tail_sec_dict[key] = rui[key][tail_sec_time[0]:tail_sec_time[1] + 1]

    if weldstage_1_dict['is'] == True and weldstage_2_dict['is'] == True:
        if weldstage_1_dict['time_section'][1] + 1 < weldstage_2_dict['time_section'][0] - 1:
            cool_1_dict['is'] = True
            cool_1_dict['time_section'] = [weldstage_1_dict['time_section'][1] + 1,
                                           weldstage_2_dict['time_section'][0] - 1]
            cool_1_dict['time'] = cool_1_dict['time_section'][1] - cool_1_dict['time_section'][0] + 1

    if weldstage_2_dict['is'] == True and weldstage_3_dict['is'] == True:
        if weldstage_2_dict['time_section'][1] + 1 < weldstage_3_dict['time_section'][0] - 1:
            cool_2_dict['is'] = True
            cool_2_dict['time_section'] = [weldstage_2_dict['time_section'][1] + 1,
                                           weldstage_3_dict['time_section'][0] - 1]
            cool_2_dict['time'] = cool_2_dict['time_section'][1] - cool_2_dict['time_section'][0] + 1

    return [total_weld_dict, weldstage_1_dict, cool_1_dict, weldstage_2_dict, cool_2_dict, weldstage_3_dict,
            tail_sec_dict]


def bsn_rui_smooth(rui, length=1000):
    sections = get_weld_sections(rui)
    new_rui = {
        'I': [],
        'U': [],
        'R': [],
        'P': [],
    }

    if sections[1]['is']:
        for key in ['I', 'U', 'R', 'P']:
            sec_sm = bsn_rui_least_square_smooth(sections[1][key])
            sec_sm = bsn_rui_least_square_smooth(sec_sm)
            sections[1][key] = sec_sm
            new_rui[key].extend(sec_sm)

    if sections[2]['is']:
        for key in ['I', 'U', 'R', 'P']:
            new_rui[key].extend([0. for _ in range(sections[2]['time'])])

    if sections[3]['is']:
        for key in ['I', 'U', 'R', 'P']:
            sec_sm = bsn_rui_least_square_smooth(sections[3][key])
            sec_sm = bsn_rui_least_square_smooth(sec_sm)
            sections[3][key] = sec_sm
            new_rui[key].extend(sec_sm)

    if sections[4]['is']:
        for key in ['I', 'U', 'R', 'P']:
            new_rui[key].extend([0. for _ in range(sections[4]['time'])])

    if sections[5]['is']:
        for key in ['I', 'U', 'R', 'P']:
            sec_sm = bsn_rui_least_square_smooth(sections[5][key])
            sec_sm = bsn_rui_least_square_smooth(sec_sm)
            sections[5][key] = sec_sm
            new_rui[key].extend(sec_sm)
    remaining_len = length - len(new_rui['I'])
    for key in ['I', 'U', 'R', 'P']:
        new_rui[key].extend([0. for _ in range(remaining_len)])

    return new_rui, sections


def level2onehot(level):
    sign = 0
    if level > 0:
        sign = 1
    elif level < 0:
        sign = -1
    if level == 0:
        return [1, 0, 0]
    if abs(level) == 1:
        return [0, 1 * sign, 0]
    if abs(level) == 2:
        return [0, 0, 1 * sign]


def find_section_splash(R, is_have_pre_sec=False, time_head_tail=10, time_min_thr=2, time_max_thr=50, r_slope_thr=2e-4):
    # R = section['R']
    # I = section['I']
    # U = section['U']
    # P = section['P']

    '''
    Step 1, 计算R的细节特征
    '''
    # 合并斜率等级是-2和-1的区，并重新标记为-1
    r_deriv_1, r_deriv_2 = curve_diff(R)
    r_deriv_1_groups, r_deriv_1_sections = find_firstderiv_sections_base_sec(r_deriv_1)
    deriv_sections = r_deriv_1_sections
    len_secs = len(deriv_sections)
    level_ls = []
    for ii in range(len_secs):  # 前后处理
        item = deriv_sections[ii]
        this_sec, this_lev = item
        this_c = this_sec[1] - this_sec[0]
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
        if this_c < 3:
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
    # 判断是否发生飞溅
    splash_times = 0
    start_t1 = 0
    end_t1 = 0
    delta_r1 = 0
    slope_r1 = 0
    start_t2 = 0
    end_t2 = 0
    delta_r2 = 0
    slope_r2 = 0
    last_time = new_deriv_secs[-1][0][1]
    len_secs = len(new_deriv_secs)
    for ii in range(1, len_secs):
        sec = new_deriv_secs[ii]
        x_sec, level = sec
        if level < 0 and x_sec[0] > time_head_tail:
            pre_sec = new_deriv_secs[ii - 1]
            if pre_sec[1] == 1:
                continue
            if pre_sec[0][1] - pre_sec[0][0] < 20:
                if ii - 2 >= 0:
                    pre_sec = new_deriv_secs[ii - 2]
                    if pre_sec[1] == 1:
                        continue

            r_delta = abs(float(R[x_sec[0]]) - float(R[x_sec[1]]))  # 实际电阻降的绝对值
            t_duration = float(x_sec[1] - x_sec[0])  # 单位ms
            r_slope = r_delta / t_duration

            if r_slope >= r_slope_thr and \
                    last_time - x_sec[0] > time_head_tail and \
                    t_duration > time_min_thr and t_duration < time_max_thr:
                tmp_sec = R[x_sec[1] + 1:(x_sec[1] + last_time - time_head_tail)]
                if isinstance(tmp_sec, (list, tuple)):
                    len_tmp = len(tmp_sec)
                elif isinstance(tmp_sec, np.ndarray):
                    len_tmp = tmp_sec.shape[0]
                else:
                    len_tmp = 0
                if len_tmp > 0:
                    maybe_splash = R[x_sec[0]]
                    max_tmp_sec = np.max(tmp_sec)
                    max_idx = int(np.where(tmp_sec == np.max(tmp_sec))[0][0])
                    if (maybe_splash > max_tmp_sec or max_idx > 30) and r_delta > delta_r1:
                        if splash_times == 0:
                            splash_times = 1
                            start_t1 = x_sec[0]
                            end_t1 = x_sec[1]
                            delta_r1 = r_delta
                            slope_r1 = r_slope
                        elif splash_times == 1:
                            splash_times = 2
                            start_t2 = x_sec[0]
                            end_t2 = x_sec[1]
                            delta_r2 = r_delta
                            slope_r2 = r_slope
    return {'splash_times': splash_times, 'start_t1': start_t1, 'end_t1': end_t1, 'delta_r1': delta_r1,
            'slope_r1': slope_r1, 'start_t2': start_t2, 'end_t2': end_t2, 'delta_r2': delta_r2,
            'slope_r2': slope_r2}


def find_section_splash_2(curve, is_have_pre_sec=False):
    # if isinstance(curve,np.ndarray):
    #     len_curve = curve.shape[0]
    # elif isinstance(curve,(list,tuple)):
    #     len_curve = len(curve)
    # num_thr = int(len_curve*0.1)
    # diff_max = np.max(diff1)
    # diff_min = np.min(diff1)
    # num_his = int((diff_max - diff_min) / 0.0002)
    # if num_his < 1:
    #     num_his = 1
    # hist, bins = np.histogram(diff1, num_his)
    # selected_bins_idx = []  # 记录直方图中符合要求的bin的位置
    # idx_thr = 0
    # hist_max = np.max(hist)
    # if num_thr >= hist_max:
    #     while num_thr >= hist_max:
    #         num_thr -= 5
    #     if num_thr < 2:
    #         num_thr = 2
    # for col in hist:  # 遍历直方图中所有柱（列）
    #     if col > num_thr: # 判断每列的数量是否超过阈值
    #         if idx_thr not in selected_bins_idx:
    #             selected_bins_idx.append(idx_thr)
    #         if idx_thr + 1 not in selected_bins_idx:
    #             selected_bins_idx.append(idx_thr + 1)
    #     idx_thr += 1
    # selected_bins = bins[selected_bins_idx]  # 直方图中符合要求的bin的边界值
    # min_ = np.min(selected_bins)
    # max_ = np.max(selected_bins)
    # gte_idxs = np.where(curve >= min_)[0].tolist()
    # lte_idxs = np.where(curve <= max_)[0].tolist()
    # selected_i_ls = list(set(gte_idxs) & set(lte_idxs))
    # selected_i_ls.sort()
    #################################333
    diff1, diff2 = curve_diff(curve)
    time_coords = np.where(diff1 < -0.0007)[0].tolist()
    time_groups, len_groups = find_coord_continuous_section(time_coords)
    splash_sections = []
    if is_have_pre_sec:
        limit = 25
    else:
        limit = 50
    for sec in time_groups:
        start, end = sec
        if start <= limit:
            start = min([limit + 1, end])
        length = end - start + 1
        is_ok = True
        newsec = [start, end]
        if length > 30:
            splash_diff = diff1[start:end + 1]
            diff_max = np.max(splash_diff)
            diff_min = np.min(splash_diff)
            num_his = 10
            hist, bins = np.histogram(splash_diff, num_his)
            sort_hist = hist.copy()
            sort_hist = abs(np.sort(-sort_hist))

            hist_thr = sort_hist[3]
            idx_list = np.argwhere(hist >= hist_thr).squeeze().tolist()
            peak_sections, _ = find_coord_continuous_section(idx_list, is_single=True)
            if len(peak_sections) <= 1:  # 只有单峰，过长的稳定下降段，是被误检的飞溅段
                is_ok = False
            else:
                is_ok = True
                first_hist_sec = hist[peak_sections[0][0]:peak_sections[0][1] + 1]
                first_max_idx = np.argwhere(first_hist_sec == np.max(first_hist_sec))[0][0] + peak_sections[0][0]
                second_hist_sec = hist[peak_sections[1][0]:peak_sections[1][1] + 1]
                second_max_idx = np.argwhere(second_hist_sec == np.max(second_hist_sec))[0][0] + peak_sections[1][0]
                first_to_second = hist[first_max_idx:second_max_idx + 1]
                min_first_to_second_idx = np.argwhere(first_to_second == np.min(first_to_second))[0][0] + first_max_idx
                selected_bins = bins[0:min_first_to_second_idx + 1]
                min_bin = np.min(selected_bins)
                max_bin = np.max(selected_bins)
                gte_idxs = np.argwhere(splash_diff >= min_bin).squeeze().tolist()
                lte_idxs = np.argwhere(splash_diff <= max_bin).squeeze().tolist()
                selected_idxs = list(set(gte_idxs) & set(lte_idxs))
                selected_idxs.sort()
                newsecs, newsecs_len_ls = find_coord_continuous_section(selected_idxs, is_single=False)
                selected_sec_idx = np.argwhere(np.array(newsecs_len_ls) == np.max(newsecs_len_ls))[0][0]
                newsec = newsecs[selected_sec_idx]
                newsec = [start + newsec[0], start + newsec[1]]
        if is_ok and start > limit and start < len(curve) - 10 and length > 2:
            splash_sections.append(newsec)

    for ii in range(len(splash_sections) - 1, -1, -1):
        start_min = np.min(diff2[splash_sections[ii][0] - 5:splash_sections[ii][0] + 6])
        end_max = np.max(diff2[splash_sections[ii][1] - 5:splash_sections[ii][1] + 6])
        if not (start_min <= -0.00008 and end_max >= 0.00008):
            splash_sections.pop(ii)

    # 判断是否发生飞溅
    splash_times = len(splash_sections)
    start_t1 = 0
    end_t1 = 0
    delta_r1 = 0
    slope_r1 = 0
    start_t2 = 0
    end_t2 = 0
    delta_r2 = 0
    slope_r2 = 0

    if len(splash_sections) > 0:
        start_t1 = splash_sections[0][0]
        end_t1 = splash_sections[0][1]
        delta_r1 = abs(curve[end_t1] - curve[start_t1])
        slope_r1 = delta_r1 / (end_t1 - start_t1 + 1)
        if len(splash_sections) > 1:
            start_t2 = splash_sections[1][0]
            end_t2 = splash_sections[1][1]
            delta_r2 = abs(curve[end_t2] - curve[start_t2])
            slope_r2 = delta_r1 / (end_t2 - start_t2 + 1)

    return {'splash_times': splash_times, 'start_t1': start_t1, 'end_t1': end_t1, 'delta_r1': delta_r1,
            'slope_r1': slope_r1, 'start_t2': start_t2, 'end_t2': end_t2, 'delta_r2': delta_r2,
            'slope_r2': slope_r2}


def find_rui_splash(sections, func=find_section_splash_2, is_preweld=False):
    sp1_sec = sections[1]
    cool1_sec = sections[2]
    sp2_sec = sections[3]
    cool2_sec = sections[4]
    sp3_sec = sections[5]
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    is_have_pre_sec = False
    if sp1_sec['is']:
        t1 = sp1_sec["time"]
    if cool1_sec['is']:
        t2 = cool1_sec["time"]
    if sp2_sec['is']:
        t3 = sp2_sec["time"]
    if cool2_sec['is']:
        t4 = cool2_sec["time"]
    pre_time = t1 + t2
    main_time = pre_time + t3 + t4
    if t1 > 0:
        is_have_pre_sec = True

    splash_times = 0
    start_t1 = 0
    end_t1 = 0
    delta_r1 = 0
    slope_r1 = 0
    start_t2 = 0
    end_t2 = 0
    delta_r2 = 0
    slope_r2 = 0

    if is_preweld:
        if sp1_sec['is'] and t1 > 50:
            splash_1 = func(sp1_sec['R'])
            if splash_1['splash_times'] > 0 and splash_1['start_t1'] > 50:
                splash_times = splash_1['splash_times']
                start_t1 = splash_1['start_t1']
                end_t1 = splash_1['end_t1']
                delta_r1 = splash_1['delta_r1']
                slope_r1 = splash_1['slope_r1']
                if splash_1['splash_times'] > 1:
                    start_t2 = splash_1['start_t2']
                    end_t2 = splash_1['end_t2']
                    delta_r2 = splash_1['delta_r2']
                    slope_r2 = splash_1['slope_r2']

    if sp2_sec['is']:
        splash_2 = func(sp2_sec['R'], is_have_pre_sec=is_have_pre_sec)
        if splash_2['splash_times'] > 0:
            splash_times = splash_2['splash_times']
            start_t1 = splash_2['start_t1'] + pre_time
            end_t1 = splash_2['end_t1'] + pre_time
            delta_r1 = splash_2['delta_r1']
            slope_r1 = splash_2['slope_r1']
            if splash_2['splash_times'] > 1:
                start_t2 = splash_2['start_t2'] + pre_time
                end_t2 = splash_2['end_t2'] + pre_time
                delta_r2 = splash_2['delta_r2']
                slope_r2 = splash_2['slope_r2']
    if splash_times < 2:
        if sp3_sec['is']:
            splash_3 = func(sp3_sec['R'], is_have_pre_sec=is_have_pre_sec)
            if splash_times < 1:
                splash_times = splash_3['splash_times']
                if splash_3['splash_times'] > 0:
                    start_t1 = splash_3['start_t1'] + main_time
                    end_t1 = splash_3['end_t1'] + main_time
                    delta_r1 = splash_3['delta_r1']
                    slope_r1 = splash_3['slope_r1']
                    if splash_3['splash_times'] > 1:
                        start_t2 = splash_3['start_t2'] + main_time
                        end_t2 = splash_3['end_t2'] + main_time
                        delta_r2 = splash_3['delta_r2']
                        slope_r2 = splash_3['slope_r2']
            elif splash_times == 1 and splash_3['splash_times'] > 0:
                splash_times += 1
                start_t2 = splash_3['start_t1'] + main_time
                end_t2 = splash_3['end_t1'] + main_time
                delta_r2 = splash_3['delta_r1']
                slope_r2 = splash_3['slope_r1']
    index = 1 if delta_r1 >= delta_r2 else 2
    start_t = start_t1 if index == 1 else start_t2
    end_t = end_t1 if index == 1 else end_t2
    delta_r = delta_r1 if index == 1 else delta_r2
    slope_r = slope_r1 if index == 1 else slope_r2
    splash_score = int((delta_r / 0.006) * 100)
    splash_score = 100 if splash_score > 100 else splash_score
    return {'splash_times': splash_times, 'splash_score': splash_score,
            'start_t': start_t, 'end_t': end_t, 'delta_r': delta_r, 'slope_r': slope_r,
            'start_t1': start_t1, 'end_t1': end_t1, 'delta_r1': delta_r1, 'slope_r1': slope_r1,
            'start_t2': start_t2, 'end_t2': end_t2, 'delta_r2': delta_r2, 'slope_r2': slope_r2}


def cal_sp_section_settings_params(i_sec,
                                   his_interval=0.2,
                                   is_check_upslope=True,
                                   upslope_t_thr=100,
                                   upslope_i_thr=1.,
                                   num_iparam_thr=40,
                                   upslope_t_min_length=20):
    '''
    计算某段RUI曲线的设置参数。time_sections记录了焊接1（预焊）、焊接2（主焊）、焊接3（回火或脉冲数等于2的第2段）
    '''
    '''
    # 设置参数
    main_ka, main_ms, upslope_ka, upslope_ms
    '''

    if not isinstance(i_sec, np.ndarray):
        if isinstance(i_sec, (list, tuple)):
            i_sec = np.array(i_sec)
        else:
            raise Exception('The type of i_sec is in [np.ndarray, list, tuple].')

    main_ka = 0.
    main_ms = 0
    upslope_ka = 0.
    upslope_ms = 0
    is_upslope = False

    i_min = np.min(i_sec)
    i_max = np.max(i_sec)

    # 利用直方图计算电流的均值
    num_his = int((i_max - i_min) / his_interval)
    if num_his < 1:
        num_his = 1
    hist, bins = np.histogram(i_sec, num_his)
    selected_bins_idx = []  # 记录直方图中符合要求的bin的位置
    idx_thr = 0
    hist_max = np.max(hist)
    if num_iparam_thr >= hist_max:
        while num_iparam_thr >= hist_max:
            num_iparam_thr -= 5
        if num_iparam_thr < 2:
            num_iparam_thr = 2
    for col in hist:  # 遍历直方图中所有柱（列）
        if col > num_iparam_thr:  # 判断每列的数量是否超过阈值
            if idx_thr not in selected_bins_idx:
                selected_bins_idx.append(idx_thr)
            if idx_thr + 1 not in selected_bins_idx:
                selected_bins_idx.append(idx_thr + 1)
        idx_thr += 1
    selected_bins = bins[selected_bins_idx]  # 直方图中符合要求的bin的边界值
    min_ = math.floor(np.min(selected_bins) * 10)
    i_param_min = min_ / 10
    max_ = math.floor(np.max(selected_bins) * 10)
    i_param_max = 0.1 + (max_ / 10)
    idx_min = np.where(i_sec == np.min(i_sec))[0][0]  # 最小值位置
    i_gte_idxs = np.where(i_sec >= i_param_min)[0].tolist()
    i_lte_idxs = np.where(i_sec <= i_param_max)[0].tolist()
    selected_i_ls = list(set(i_gte_idxs) & set(i_lte_idxs))
    selected_i_ls.sort()
    if is_check_upslope:
        if idx_min < upslope_t_thr:
            for i in range(len(selected_i_ls) - 1, -1, -1):
                if selected_i_ls[i] < idx_min:
                    selected_i_ls.pop(i)
    selected_i = i_sec[selected_i_ls]  # 符合范围内的电流值
    i_mean = round(np.mean(selected_i), 1)

    if is_check_upslope:
        i_gte_idxs = np.where(i_sec >= selected_bins[0])[0].tolist()
        i_lte_idxs = np.where(i_sec <= selected_bins[1])[0].tolist()
        selected_i_ls = list(set(i_gte_idxs) & set(i_lte_idxs))
        selected_i_ls.sort()
        if idx_min < upslope_t_thr:
            for i in range(len(selected_i_ls) - 1, -1, -1):
                if selected_i_ls[i] < idx_min:
                    selected_i_ls.pop(i)
        selected_i = i_sec[selected_i_ls]  # 符合范围内的电流值
        # first_thr = (selected_bins[0] + selected_bins[1]) / 2
        first_thr = np.mean(selected_i)
        first_thr = int(first_thr * 100) / 100
        for ii in selected_i_ls:
            if i_sec[ii] >= first_thr:
                idx_main = ii
                break
        if idx_min < upslope_t_thr and i_max - i_min > upslope_i_thr and idx_main - idx_min > upslope_t_min_length:
            is_upslope = True
            var_1 = idx_main / 5
            var_1 = round(var_1)
            upslope_ka = round(i_sec[idx_min], 1)
            upslope_ms = int(var_1 * 5)

    tmp_time_length = i_sec.shape[0]
    var_1 = tmp_time_length / 5
    var_1 = round(var_1)
    main_ms = int(var_1 * 5)
    main_ka = i_mean

    return {'main_ka': main_ka, 'main_ms': main_ms,
            'upslope_ka': upslope_ka, 'upslope_ms': upslope_ms,
            'is_upslope': is_upslope}


def cal_sp_section_features(section, num_sections=8):
    '''
    计算某段RUI曲线的细节特征。time_sections记录了焊接1（预焊）、焊接2（主焊）、焊接3（回火或脉冲数等于2的第2段）
    '''
    if section['is'] == True and section['is_cur'] == True:
        R = section['R']
        I = section['I']
        U = section['U']
        P = section['P']

        '''
        Step 1, 计算R的细节特征
        '''
        # 合并斜率等级是-2和-1的区，并重新标记为-1
        r_deriv_1, r_deriv_2 = curve_diff(R)
        r_deriv_1_groups, r_deriv_1_sections = find_firstderiv_sections_base_sec(r_deriv_1)
        deriv_sections = r_deriv_1_sections
        len_secs = len(deriv_sections)
        level_ls = []
        for ii in range(len_secs):  # 前后处理
            item = deriv_sections[ii]
            this_sec, this_lev = item
            this_c = this_sec[1] - this_sec[0]
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
            if this_c < 3:
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

        # 定义特征向量
        features = []
        num_valid = 0

        # 取前num_sections个区，保证向量长度一致。
        new_deriv_secs_result = new_deriv_secs
        if len(new_deriv_secs) > num_sections:
            new_deriv_secs_result = new_deriv_secs[0:num_sections]
            new_deriv_secs_result[-1][0][1] = new_deriv_secs[-1][0][1]

        # 主要特征
        for i in range(num_sections):
            if i < len(new_deriv_secs_result):
                [x_sec, level] = new_deriv_secs_result[i]
                level_oh = level2onehot(level)  # 斜率等级独热编码
                sec_head = float(R[x_sec[0]])  # 起始值
                sec_end = float(R[x_sec[1]])  # 终止值
                sec_curve = R[x_sec[0]:x_sec[1]]
                x_min = int(np.where(np.min(sec_curve))[0][0]) + x_sec[0]
                x_max = int(np.where(np.max(sec_curve))[0][0]) + x_sec[0]
                sec_min = float(np.min(sec_curve))  # 最小值
                sec_max = float(np.max(sec_curve))  # 最大值

                features.extend(level_oh)  # 导数分区
                features.append(x_sec[0] / 1000)  # 起始时间
                features.append((x_sec[1]) / 1000)  # 终止时间
                features.append((x_min) / 1000)  # 最小值时间
                features.append((x_max) / 1000)  # 最大值时间
                features.append(sec_head)  # 起始值
                features.append(sec_end)  # 终止值
                features.append(sec_min)  # 最小值
                features.append(sec_max)  # 最大值的差值

                num_valid += 1
            else:
                features.extend([0. for _ in range(11)])
        features.append(num_valid)

        '''
        Step 2, 计算R, U, I, P的两段特征
        '''
        # 曲线局部特征区间提取的特征, 104维
        local_section_feats = []
        first_big_sec = []
        second_big_sec = []

        num_sec = len(deriv_sections)
        is_first = True
        for ii in range(num_sec):
            sec = deriv_sections[ii]
            [x_sec, level] = sec
            if len(first_big_sec) == 0:
                first_big_sec = [[x_sec[0], x_sec[1]], level]
            else:
                if x_sec[1] > 60 and level == 0:
                    is_first = False
                if is_first:
                    first_big_sec[0][1], first_big_sec[1] = x_sec[1], level
                else:
                    if len(second_big_sec) == 0:
                        second_big_sec = [[x_sec[0], x_sec[1]], level]
                    else:
                        second_big_sec[0][1], second_big_sec[1] = x_sec[1], level
        if len(second_big_sec) == 0:
            second_big_sec = [[0, 0], 0]
        first_feats = get_rui_simple_feats_3(first_big_sec, R, U, I, P)
        second_feats = get_rui_simple_feats_3(second_big_sec, R, U, I, P)
        local_section_feats.extend(first_feats)
        local_section_feats.extend(second_feats)
        features.extend(local_section_feats)
        return features
    else:
        return [0. for _ in range(num_sections * 11 + 1 + 81 * 2)]


def cal_sp_pre_weld_feats(section):
    if section['is'] == True and section['is_cur'] == True:
        # 去除第1位置和最后一个位置的0值。
        R = section['R']
        I = section['I']
        U = section['U']
        P = section['P']

        # 曲线全局特征, 52维
        global_feats = get_rui_simple_feats_2_2(R, U, I, P)
        return global_feats
    else:
        return [0. for _ in range(98)]


def cal_sp_settings_params(sections):
    '''
    计算某段RUI曲线的设置参数。time_sections记录了焊接1（预焊）、焊接2（主焊）、焊接3（回火或脉冲数等于2的第2段）
    '''
    '''
    # 设置参数
    焊接1：sp_1_ka, sp_1_weld_ms, sp_1_cool_ms
    上坡：sp_init_upslope_ka, sp_upst_ms
    焊接2：sp_2_ka, sp_2_weld_ms, sp_2_cool_ms
    焊接3：sp_3_ka, sp_3_weld_ms
    '''
    sp_1_ka, sp_1_weld_ms, sp_1_cool_ms = 0., 0, 0
    sp_init_upslope_ka, sp_upst_ms = 0., 0
    sp_2_ka, sp_2_weld_ms, sp_2_cool_ms = 0., 0, 0
    sp_3_ka, sp_3_weld_ms = 0., 0

    sp1_sec = sections[1]
    cool1_sec = sections[2]
    sp2_sec = sections[3]
    cool2_sec = sections[4]
    sp3_sec = sections[5]

    if sp1_sec['is']:
        i_sec = sp1_sec['I']
        params_dict = cal_sp_section_settings_params(i_sec,
                                                     his_interval=0.2,
                                                     is_check_upslope=False,
                                                     upslope_t_thr=100,
                                                     upslope_i_thr=1.2,
                                                     num_iparam_thr=40,
                                                     upslope_t_min_length=20)
        # 设置参数
        sp_1_ka = params_dict['main_ka']  # 焊接1设置电流
        sp_1_weld_ms = params_dict['main_ms']  # 焊接1设置时间

    if cool1_sec['is']:
        tmp_time_length = cool1_sec['time_section'][1] - cool1_sec['time_section'][0]
        var_1 = tmp_time_length / 5
        var_1 = round(var_1)
        # 设置参数
        sp_1_cool_ms = int(var_1 * 5)  # 焊接1设置冷却时间
    if sp2_sec['is']:
        i_sec = sp2_sec['I']
        params_dict = cal_sp_section_settings_params(i_sec,
                                                     his_interval=0.2,
                                                     is_check_upslope=True,
                                                     upslope_t_thr=100,
                                                     upslope_i_thr=1.2,
                                                     num_iparam_thr=40,
                                                     upslope_t_min_length=20)
        # 设置参数
        sp_2_ka = params_dict['main_ka']  # 焊接2设置电流
        sp_2_weld_ms = params_dict['main_ms']  # 焊接2设置时间
        if params_dict['is_upslope']:
            sp_init_upslope_ka, sp_upst_ms = params_dict['upslope_ka'], params_dict['upslope_ms']  # 上坡电流和上坡时间
    if cool2_sec['is']:
        tmp_time_length = cool2_sec['time_section'][1] - cool2_sec['time_section'][0]
        var_1 = tmp_time_length / 5
        var_1 = round(var_1)
        # 设置参数
        sp_2_cool_ms = int(var_1 * 5)  # 焊接2设置冷却时间
    if sp3_sec['is']:
        i_sec = sp3_sec['I']
        params_dict = cal_sp_section_settings_params(i_sec,
                                                     his_interval=0.2,
                                                     is_check_upslope=False,
                                                     upslope_t_thr=100,
                                                     upslope_i_thr=1.2,
                                                     num_iparam_thr=40,
                                                     upslope_t_min_length=20)
        # 设置参数
        sp_3_ka = params_dict['main_ka']  # 焊接3设置电流
        sp_3_weld_ms = params_dict['main_ms']  # 焊接3设置时间
    return [
        sp_1_ka, sp_1_weld_ms, sp_1_cool_ms,
        sp_init_upslope_ka, sp_upst_ms,
        sp_2_ka, sp_2_weld_ms, sp_2_cool_ms,
        sp_3_ka, sp_3_weld_ms,
    ], {
        'sp_1_ka': sp_1_ka, 'sp_1_weld_ms': sp_1_weld_ms, 'sp_1_cool_ms': sp_1_cool_ms,
        'sp_init_upslope_ka': sp_init_upslope_ka, 'sp_upst_ms': sp_upst_ms,
        'sp_2_ka': sp_2_ka, 'sp_2_weld_ms': sp_2_weld_ms, 'sp_2_cool_ms': sp_2_cool_ms,
        'sp_3_ka': sp_3_ka, 'sp_3_weld_ms': sp_3_weld_ms,
    }


def cal_sp_actual_params(sections):
    '''
    计算某段RUI曲线的实际参数。time_sections记录了焊接1（预焊）、焊接2（主焊）、焊接3（回火或脉冲数等于2的第2段）
    '''
    '''
    # 实际参数
    sp_1_i, sp_1_u, sp_1_r, sp_1_p, sp_1_q, sp_1_t, sp_1_cool_t
    sp_2_i, sp_2_u, sp_2_r, sp_2_p, sp_2_q, sp_2_t, sp_2_cool_t
    sp_3_i, sp_3_u, sp_3_r, sp_3_p, sp_3_q, sp_3_t
    sp_total_i, sp_total_u, sp_total_r, sp_total_p, sp_total_q, sp_total_t
    '''
    sp_1_i, sp_1_u, sp_1_r, sp_1_p, sp_1_q, sp_1_t, sp_1_cool_t = 0., 0., 0., 0., 0., 0, 0
    sp_2_i, sp_2_u, sp_2_r, sp_2_p, sp_2_q, sp_2_t, sp_2_cool_t = 0., 0., 0., 0., 0., 0, 0
    sp_3_i, sp_3_u, sp_3_r, sp_3_p, sp_3_q, sp_3_t = 0., 0., 0., 0., 0., 0

    sp1_sec = sections[1]
    cool1_sec = sections[2]
    sp2_sec = sections[3]
    cool2_sec = sections[4]
    sp3_sec = sections[5]

    if sp1_sec['is']:
        # 实际参数
        sp_1_act_val = get_rui_sec_actual_parmas(R=sp1_sec['R'], U=sp1_sec['U'], I=sp1_sec['I'], P=sp1_sec['P'])
        sp_1_i, sp_1_u, sp_1_r, sp_1_p, sp_1_q, sp_1_t = sp_1_act_val['sp_i'], sp_1_act_val['sp_u'], sp_1_act_val[
            'sp_r'], sp_1_act_val['sp_p'], sp_1_act_val['sp_q'], sp_1_act_val['sp_t']

    if cool1_sec['is']:
        tmp_time_length = cool1_sec['time_section'][1] - cool1_sec['time_section'][0]
        # 实际参数
        sp_1_cool_t = tmp_time_length  # 焊接1实际冷却时间
    if sp2_sec['is']:
        # 实际参数
        sp_2_act_val = get_rui_sec_actual_parmas(R=sp2_sec['R'], U=sp2_sec['U'], I=sp2_sec['I'], P=sp2_sec['P'])
        sp_2_i, sp_2_u, sp_2_r, sp_2_p, sp_2_q, sp_2_t = sp_2_act_val['sp_i'], sp_2_act_val['sp_u'], sp_2_act_val[
            'sp_r'], sp_2_act_val['sp_p'], sp_2_act_val['sp_q'], sp_2_act_val['sp_t']
    if cool2_sec['is']:
        tmp_time_length = cool2_sec['time_section'][1] - cool2_sec['time_section'][0]
        # 实际参数
        sp_2_cool_t = tmp_time_length  # 焊接2实际冷却时间
    if sp3_sec['is']:
        # 实际参数
        sp_3_act_val = get_rui_sec_actual_parmas(R=sp3_sec['R'], U=sp3_sec['U'], I=sp3_sec['I'], P=sp3_sec['P'])
        sp_3_i, sp_3_u, sp_3_r, sp_3_p, sp_3_q, sp_3_t = sp_3_act_val['sp_i'], sp_3_act_val['sp_u'], sp_3_act_val[
            'sp_r'], sp_3_act_val['sp_p'], sp_3_act_val['sp_q'], sp_3_act_val['sp_t']
    return [
        sp_1_i, sp_1_u, sp_1_r, sp_1_p, sp_1_q, sp_1_t, sp_1_cool_t,
        sp_2_i, sp_2_u, sp_2_r, sp_2_p, sp_2_q, sp_2_t, sp_2_cool_t,
        sp_3_i, sp_3_u, sp_3_r, sp_3_p, sp_3_q, sp_3_t,
    ], {
        'sp_1_i': sp_1_i, 'sp_1_u': sp_1_u, 'sp_1_r': sp_1_r, 'sp_1_p': sp_1_p,
        'sp_1_q': sp_1_q, 'sp_1_t': sp_1_t, 'sp_1_cool_t': sp_1_cool_t,
        'sp_2_i': sp_2_i, 'sp_2_u': sp_2_u, 'sp_2_r': sp_2_r, 'sp_2_p': sp_2_p,
        'sp_2_q': sp_2_q, 'sp_2_t': sp_2_t, 'sp_2_cool_t': sp_2_cool_t,
        'sp_3_i': sp_3_i, 'sp_3_u': sp_3_u, 'sp_3_r': sp_3_r, 'sp_3_p': sp_3_p, 'sp_3_q': sp_3_q,
        'sp_3_t': sp_3_t,
    }


def cal_sp_features(sections):
    '''
    计算RUI曲线的细节特征。
    '''
    sp1_sec = sections[1]
    cool1_sec = sections[2]
    sp2_sec = sections[3]
    cool2_sec = sections[4]
    sp3_sec = sections[5]

    sp1_feats = cal_sp_pre_weld_feats(sp1_sec)
    sp2_feats = cal_sp_section_features(sp2_sec, num_sections=8)
    sp3_feats = cal_sp_section_features(sp3_sec, num_sections=4)

    result = []
    # result.extend(sp1_feats)
    result.extend(sp2_feats)
    # result.extend(sp3_feats)

    return result


def get_params_from_rui(rui):
    R = rui['R']
    I = rui['I']
    U = rui['U']
    P = rui['P']
    new_rui, sections = bsn_rui_smooth(rui)
    params_ls, params_dict = cal_sp_settings_params(sections)

    total_act_dict = get_rui_sec_actual_parmas(R, U, I, P)
    total_act_ls = [total_act_dict['sp_i'], total_act_dict['sp_u'], total_act_dict['sp_r'], total_act_dict['sp_p'],
                    total_act_dict['sp_q'], total_act_dict['sp_t']]
    actual_val_ls, actual_val_dict = cal_sp_actual_params(sections)
    detail_features = cal_sp_features(sections)

    result_ls = []
    result_dict = {}
    result_ls.extend(params_ls)
    result_ls.extend(total_act_ls)
    result_ls.extend(actual_val_ls)
    result_ls.extend(detail_features)

    result_dict['smoothed_curve'] = new_rui
    result_dict['settings_params'] = params_dict
    result_dict['total_actual_vals'] = total_act_dict
    result_dict['section_actual_vals'] = actual_val_dict
    result_dict['detail_features'] = detail_features

    return result_ls, result_dict


def is_params_consistent(rui_params, ref_params, i_1_thr=0.5, i_2_thr=1, t_thr=5):
    '''
    返回值：第1个bool表示最典型的判别信息是否一致，不一致则无需关心第二个bool。
    第2个bool表示次典型的判别信息是否一致，表示在最典型信息一致的情况下判断其他信息
    第3个是主焊阶段的时间，
    '''
    sp1ka_res = abs(rui_params['sp_1_ka'] - ref_params['sp_1_ka'])
    sp1ms_res = abs(rui_params['sp_1_weld_ms'] - ref_params['sp_1_weld_ms'])
    sp1coolms_res = abs(rui_params['sp_1_cool_ms'] - ref_params['sp_1_cool_ms'])
    sp2ka_res = abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka'])
    sp2ms_res = abs(rui_params['sp_2_weld_ms'] - ref_params['sp_2_weld_ms'])
    sp2coolms_res = abs(rui_params['sp_2_cool_ms'] - ref_params['sp_2_cool_ms'])
    sp3ka_res = abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka'])
    sp3ms_res = abs(rui_params['sp_3_weld_ms'] - ref_params['sp_3_weld_ms'])
    res = (sp3ms_res + 10 * sp2ms_res + sp1ms_res + sp1coolms_res + sp2coolms_res) / 1000 + (
            sp3ka_res + 10 * sp2ka_res + sp1ka_res) / 10
    # 判断最典型信息是否一致
    if abs(rui_params['sp_1_ka'] - ref_params['sp_1_ka']) > i_1_thr:
        return False, False, res
    if abs(rui_params['sp_1_weld_ms'] - ref_params['sp_1_weld_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_1_cool_ms'] - ref_params['sp_1_cool_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_2_cool_ms'] - ref_params['sp_2_cool_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka']) > i_2_thr:
        return False, False, res
    if abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka']) > i_2_thr:
        return False, False, res

    # 判断次典型信息是否一致
    if abs(rui_params['sp_2_weld_ms'] - ref_params['sp_2_weld_ms']) > t_thr:
        return True, False, res
    if abs(rui_params['sp_3_weld_ms'] - ref_params['sp_3_weld_ms']) > t_thr:
        return True, False, res
    if abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka']) > i_1_thr:
        return True, False, res
    if abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka']) > i_1_thr:
        return True, False, res
    return True, True, res


def is_rui_consistent(rui_params, ref_params, i_1_thr=0.5, i_2_thr=1, t_thr=5, r_2_thr=0.2):
    '''
    返回值：第1个bool表示最典型的判别信息是否一致，不一致则无需关心第二个bool。
    第2个bool表示次典型的判别信息是否一致，表示在最典型信息一致的情况下判断其他信息
    第3个是主焊阶段的时间，
    '''
    sp1ka_res = abs(rui_params['sp_1_ka'] - ref_params['sp_1_ka'])
    sp1ms_res = abs(rui_params['sp_1_weld_ms'] - ref_params['sp_1_weld_ms'])
    sp1coolms_res = abs(rui_params['sp_1_cool_ms'] - ref_params['sp_1_cool_ms'])
    sp2ka_res = abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka'])
    sp2ms_res = abs(rui_params['sp_2_weld_ms'] - ref_params['sp_2_weld_ms'])
    sp2coolms_res = abs(rui_params['sp_2_cool_ms'] - ref_params['sp_2_cool_ms'])
    sp3ka_res = abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka'])
    sp3ms_res = abs(rui_params['sp_3_weld_ms'] - ref_params['sp_3_weld_ms'])
    sp1r_res = abs(rui_params['sp_1_r'] - ref_params['sp_1_r'])
    sp2r_res = abs(rui_params['sp_2_r'] - ref_params['sp_2_r'])
    sp3r_res = abs(rui_params['sp_3_r'] - ref_params['sp_3_r'])
    res = (sp3ms_res + 10 * sp2ms_res + sp1ms_res + sp1coolms_res + sp2coolms_res) / 1000 + (
            sp3ka_res + 10 * sp2ka_res + sp1ka_res) / 10 + sp2r_res * 10
    # 判断最典型信息是否一致
    if abs(rui_params['sp_1_ka'] - ref_params['sp_1_ka']) > i_1_thr:
        return False, False, res
    if abs(rui_params['sp_1_weld_ms'] - ref_params['sp_1_weld_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_1_cool_ms'] - ref_params['sp_1_cool_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_2_cool_ms'] - ref_params['sp_2_cool_ms']) > t_thr:
        return False, False, res
    if abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka']) > i_2_thr:
        return False, False, res
    if abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka']) > i_2_thr:
        return False, False, res

    # 判断次典型信息是否一致
    if abs(rui_params['sp_2_weld_ms'] - ref_params['sp_2_weld_ms']) > t_thr:
        return True, False, res
    if abs(rui_params['sp_3_weld_ms'] - ref_params['sp_3_weld_ms']) > t_thr:
        return True, False, res
    if abs(rui_params['sp_2_ka'] - ref_params['sp_2_ka']) > i_1_thr:
        return True, False, res
    if abs(rui_params['sp_3_ka'] - ref_params['sp_3_ka']) > i_1_thr:
        return True, False, res
    if sp2r_res > r_2_thr:
        return True, False, res

    return True, True, res


def params_x_y_for_plot(rui_params):
    '''
    sp_1_weld_t
    sp_1_cool_t
    sp_2_weld_t
    sp_2_cool_t
    sp_3_weld_t
    '''
    time_point = 0
    # if rui_params['sp_1_weld_ms']>0:
    sp_1_weld_t = [time_point, time_point + rui_params['sp_1_weld_ms']]
    time_point += rui_params['sp_1_weld_ms']
    sp_1_cool_t = [time_point, time_point + rui_params['sp_1_cool_ms']]
    time_point += rui_params['sp_1_cool_ms']
    sp_2_weld_t = [time_point, time_point + rui_params['sp_2_weld_ms']]
    time_point += rui_params['sp_2_weld_ms']
    sp_2_cool_t = [time_point, time_point + rui_params['sp_2_cool_ms']]
    time_point += rui_params['sp_2_cool_ms']
    sp_3_weld_t = [time_point, time_point + rui_params['sp_3_weld_ms']]
    x = np.array([i for i in range(sp_3_weld_t[1])])
    y = np.zeros(sp_3_weld_t[1])
    y[sp_1_weld_t[0]:sp_1_weld_t[1]] = rui_params['sp_1_ka']
    y[sp_1_cool_t[0]:sp_1_cool_t[1]] = 0
    y[sp_2_weld_t[0]:sp_2_weld_t[1]] = rui_params['sp_2_ka']
    y[sp_2_cool_t[0]:sp_2_cool_t[1]] = 0
    y[sp_3_weld_t[0]:sp_3_weld_t[1]] = rui_params['sp_3_ka']

    return x, y


def cal_one_curve_psc(curve, ref_curve, weld_time, thr_prop=0.15):
    if not isinstance(curve, np.ndarray):
        if isinstance(curve, (list, tuple)):
            curve = np.array(curve)
        else:
            raise Exception('The type of curve is not in (np.ndarray, list, tuple).')

    if not isinstance(ref_curve, np.ndarray):
        if isinstance(ref_curve, (list, tuple)):
            ref_curve = np.array(ref_curve)
        else:
            raise Exception('The type of ref_curve is not in (np.ndarray, list, tuple).')

    rui_p = np.array(curve[0:weld_time])
    ref_p = np.array(ref_curve[0:weld_time])
    len_rui = rui_p.shape[0]
    thr_arr = ref_p * thr_prop

    residual = abs(rui_p - ref_p)

    out_bound = np.where(residual > thr_arr)[0].tolist()
    len_out_bound = len(out_bound)
    psc = 100 * (len_rui - len_out_bound) / len_rui
    psc = round(psc, 1)

    return psc


# def cal_rui_psc(rui, ref_rui):
#     _, sections = get_weld_sections(ref_rui)
#     weld_time = sections[0]['time']
#     psc_ls = []
#     for curve, thr_prop in zip(['I', 'U', 'R', 'P'], [0.051, 0.08, 0.065, 0.15]):
#         psc = cal_one_curve_psc(rui[curve], ref_rui[curve], weld_time, thr_prop=thr_prop)
#         psc_ls.append(psc)
#     return np.min(psc_ls)


def cal_psc_for_one_section(curve, ref_curve):
    psc_ls = []
    # for type_c, thr_prop, thr_min in zip(['R', 'P'], [0.065, 0.15], [-0.005, -0.5]):
    for type_c, thr_prop, thr_min in zip(['R', 'P'], [0.065, 0.15], [-0.001, -0.1]):
        rui_arr = np.array(curve[type_c])
        ref_arr = np.array(ref_curve[type_c])
        residual = rui_arr - ref_arr
        len_rui = rui_arr.shape[0]
        thr_arr = ref_arr * thr_prop

        pos_bound = np.where(residual > thr_arr)[0].tolist()
        neg_bound = np.where(residual < -1 * thr_arr)[0].tolist()
        abs_neg_bound = np.where(residual < -0.005)[0].tolist()

        len_pos = len(pos_bound)
        len_neg = len(neg_bound)
        len_abs_neg = len(abs_neg_bound)

        pos_psc = 100 * (len_rui - len_pos) / len_rui
        neg_psc = 100 * (len_rui - len_neg) / len_rui
        abs_neg = 100 * (len_rui - len_abs_neg) / len_rui

        pos_psc = round(pos_psc, 1)
        neg_psc = round(neg_psc, 1)
        abs_neg = round(abs_neg, 1)
        psc_ls.extend([pos_psc, neg_psc, abs_neg])
    return psc_ls


def cal_rui_psc(rui, ref_sections):
    psc_dict = {
        '1_r_pos': 0,
        '1_r_neg': 0,
        '1_r_abs_neg': 0,
        '1_p_pos': 0,
        '1_p_neg': 0,
        '1_p_abs_neg': 0,
        '2_r_pos': 0,
        '2_r_neg': 0,
        '2_r_abs_neg': 0,
        '2_p_pos': 0,
        '2_p_neg': 0,
        '2_p_abs_neg': 0,
        '3_r_pos': 0,
        '3_r_neg': 0,
        '3_r_abs_neg': 0,
        '3_p_pos': 0,
        '3_p_neg': 0,
        '3_p_abs_neg': 0,
    }
    sections = ref_sections
    new_sections = [sections[1], sections[3], sections[5]]
    for i in range(3):
        if new_sections[(i)]['is']:
            section = new_sections[(i)]
            time_section = section['time_section']
            rui_section = {}
            for curve in ['I', 'U', 'R', 'P']:
                rui_section[curve] = rui[curve][time_section[0]:time_section[1] + 1]
            ref_section = section
            psc_ls = cal_psc_for_one_section(rui_section, ref_section)
            psc_dict[f'{(i + 1)}_r_pos'] = psc_ls[0]
            psc_dict[f'{(i + 1)}_r_neg'] = psc_ls[1]
            psc_dict[f'{(i + 1)}_r_abs_neg'] = psc_ls[2]
            psc_dict[f'{(i + 1)}_p_pos'] = psc_ls[3]
            psc_dict[f'{(i + 1)}_p_neg'] = psc_ls[4]
            psc_dict[f'{(i + 1)}_p_abs_neg'] = psc_ls[5]

    return psc_dict


def cal_params_diff_coe(rui, ref_curve_dict):
    rui, rui_sections = bsn_rui_smooth(rui)
    _, rui_params = cal_sp_settings_params(rui_sections)
    optimal_ref = None
    min_diff = 1000000000
    for id in ref_curve_dict:
        ref_params = ref_curve_dict[id]
        sections = get_weld_sections(ref_params)
        actual_ls, actual_dict = cal_sp_actual_params(sections)
        for key in actual_dict.keys():
            ref_params[key] = actual_dict[key]
        bool1, bool2, diff_coe = is_rui_consistent(rui_params, ref_params)
        if diff_coe < min_diff:
            min_diff = diff_coe
            optimal_ref = ref_params.copy()
    ref_rui = {
        "I": optimal_ref["I"],
        "U": optimal_ref["U"],
        "R": optimal_ref["R"],
        "P": optimal_ref["P"],
    }
    ref_rui, ref_sections = bsn_rui_smooth(ref_rui)
    psc_dict = cal_rui_psc(rui, ref_sections)
    total_act_dict = get_rui_sec_actual_parmas(rui['R'], rui['U'], rui['I'], rui['P'])
    _, actual_val_dict = cal_sp_actual_params(rui_sections)
    psc_dict.update(total_act_dict)
    psc_dict.update(actual_val_dict)
    return psc_dict, rui, ref_rui
