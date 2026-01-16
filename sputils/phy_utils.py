#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from .math_utils import curve_integral, curve_simple_feats,curve_simple_feats_2


def get_mean_I(I, num_delete=10):
    if not isinstance(I, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(I, np.ndarray):
        I = I.tolist()
    if isinstance(I, tuple):
        I = list(I)

    # 去掉10个最小值
    for _ in range(num_delete):
        idx = I.index(min(I))
        I.pop(idx)

    # 去掉10个最大值
    for _ in range(num_delete):
        idx = I.index(max(I))
        I.pop(idx)
    return np.mean(I)


def get_effective_I(I, R):
    if not isinstance(I, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if not isinstance(R, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if not isinstance(I, np.ndarray):
        I = np.array(I)
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    R_integral = curve_integral(R)
    Q = curve_integral((I ** 2) * R)
    return (Q / R_integral) ** 0.5


def get_mean_U(U, num_delete=10):
    if not isinstance(U, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(U, np.ndarray):
        U = U.tolist()
    if isinstance(U, tuple):
        U = list(U)

    # 去掉10个最小值
    for _ in range(num_delete):
        idx = U.index(min(U))
        U.pop(idx)

    # 去掉10个最大值
    for _ in range(num_delete):
        idx = U.index(max(U))
        U.pop(idx)
    return np.mean(U)


def get_effective_U(U, R):
    if not isinstance(U, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if not isinstance(R, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    R_integral = curve_integral(R)
    Q = curve_integral((U ** 2) / R)
    return (Q * R_integral) ** 0.5


def get_mean_P(P, num_delete=10):
    if not isinstance(P, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(P, np.ndarray):
        P = P.tolist()
    if isinstance(P, tuple):
        P = list(P)

    # 去掉10个最小值
    for _ in range(num_delete):
        idx = P.index(min(P))
        P.pop(idx)

    # 去掉10个最大值
    for _ in range(num_delete):
        idx = P.index(max(P))
        P.pop(idx)
    return np.mean(P)


def get_Q_by_P(P):
    return curve_integral(P)


def get_rui_simple_feats(R, U, I, P):
    '''
    t, I, U, P简单特征提取
    t: 焊接段时间
    R: 起始值、终止值、最大值、最小值、平均值
    I: 起始值、终止值、最大值、最小值、平均值, 有效值
    U: 起始值、终止值、最大值、最小值、平均值, 有效值
    P: 起始值、终止值、最大值、最小值、平均值
    Q: P的积分求电热
    输出长度是39
    '''
    feats = []
    # 时间全局特征
    t_global = len(R)
    # 电阻全局
    R_x = R.copy()
    R_feats = curve_simple_feats(R_x)
    # 功率和能量的全局特征
    P_x = P.copy()
    Q = get_Q_by_P(P_x)  # 能量
    P_mean = Q / t_global  # 功率均值
    P_x = P.copy()
    P_feats = curve_simple_feats(P_x)
    # 电流全局特征
    I_x = I.copy()
    if t_global > 30:
        I_mean = get_mean_I(I_x, num_delete=10)
    else:
        I_mean = np.mean(I_x)
    I_x = I.copy()
    I_efft = get_effective_I(I_x, R_x)
    I_x = I.copy()
    I_feats = curve_simple_feats(I_x)
    # 电压全局特征
    U_x = U.copy()
    if t_global > 30:
        U_mean = get_mean_U(U_x, num_delete=2)
    else:
        U_mean = np.mean(U_x)
    U_efft = P_mean / I_efft
    U_x = U.copy()
    U_feats = curve_simple_feats(U_x)

    feats.append(t_global)  # 全局时间
    feats.extend(R_feats)  # 电阻特征
    feats.extend(P_feats)  # 电功率特征
    feats.append(P_mean)  # 电功率特殊值特征
    feats.append(Q)  # 电热全局特征
    feats.extend(I_feats)  # 电流特征
    feats.append(I_mean)  # 电流特殊值特征
    feats.append(I_efft)  # 电流特殊值特征
    feats.extend(U_feats)  # 电压特征
    feats.append(U_mean)  # 电压特殊值特征
    feats.append(U_efft)  # 电压特殊值特征
    return feats

#
# def get_rui_local_simple_feats(R, U, I, P):
#     '''
#     t, I, U, P简单特征提取
#     t: 焊接段时间
#     R: 起始值、终止值、最大值、最小值、平均值
#     I: 起始值、终止值、最大值、最小值、平均值, 有效值
#     U: 起始值、终止值、最大值、最小值、平均值, 有效值
#     P: 起始值、终止值、最大值、最小值、平均值
#     Q: P的积分求电热
#     输出长度是39
#     '''
#     feats = []
#     # 时间全局特征
#     t_global = len(R)
#     # 电阻全局
#     R_x = R.copy()
#     R_feats = curve_local_simple_feats(R_x)
#     # 功率和能量的全局特征
#     P_x = P.copy()
#     Q = get_Q_by_P(P_x)  # 能量
#     P_mean = Q / t_global  # 功率均值
#     P_x = P.copy()
#     P_feats = curve_local_simple_feats(P_x)
#     # 电流全局特征
#     I_x = I.copy()
#     if t_global > 30:
#         I_mean = get_mean_I(I_x, num_delete=10)
#     else:
#         I_mean = np.mean(I_x)
#     I_x = I.copy()
#     I_efft = get_effective_I(I_x, R_x)
#     I_x = I.copy()
#     I_feats = curve_local_simple_feats(I_x)
#     # 电压全局特征
#     U_x = U.copy()
#     if t_global > 30:
#         U_mean = get_mean_U(U_x, num_delete=2)
#     else:
#         U_mean = np.mean(U_x)
#     U_efft = P_mean / I_efft
#     U_x = U.copy()
#     U_feats = curve_local_simple_feats(U_x)
#
#     feats.append(t_global)  # 全局时间
#     feats.extend(R_feats)  # 电阻特征
#     feats.extend(P_feats)  # 电功率特征
#     feats.append(P_mean)  # 电功率特殊值特征
#     feats.append(Q)  # 电热全局特征
#     feats.extend(I_feats)  # 电流特征
#     feats.append(I_mean)  # 电流特殊值特征
#     feats.append(I_efft)  # 电流特殊值特征
#     feats.extend(U_feats)  # 电压特征
#     feats.append(U_mean)  # 电压特殊值特征
#     feats.append(U_efft)  # 电压特殊值特征
#     return feats


def get_rui_simple_feats_2(R, U, I, P):
    feats = []
    # 时间全局特征
    t_global = len(R)
    if t_global > 0:
        # 电阻全局
        R_x = R.copy()
        R_feats = curve_simple_feats(R_x)
        # 功率和能量的全局特征
        P_x = P.copy()
        Q = get_Q_by_P(P_x)  # 能量
        P_x = P.copy()
        P_feats = curve_simple_feats(P_x)
        # 电流全局特征
        I_x = I.copy()
        if t_global > 30:
            I_mean = get_mean_I(I_x, num_delete=10)
        else:
            I_mean = np.mean(I_x)
        I_x = I.copy()
        I_feats = curve_simple_feats(I_x)
        # 电压全局特征
        U_x = U.copy()
        if t_global > 30:
            U_mean = get_mean_U(U_x, num_delete=2)
        else:
            U_mean = np.mean(U_x)
        U_x = U.copy()
        U_feats = curve_simple_feats(U_x)

        feats.append(t_global)  # 全局时间
        feats.extend(R_feats)  # 电阻特征
        feats.extend(P_feats)  # 电功率特征
        feats.append(Q)  # 电热全局特征
        feats.extend(I_feats)  # 电流特征
        feats.append(I_mean)  # 电流特殊值特征
        feats.extend(U_feats)  # 电压特征
        feats.append(U_mean)  # 电压特殊值特征
    else:
        feats.extend([0. for _ in range(104)])
    return feats

def get_rui_simple_feats_2_2(R, U, I, P):

    feats = []
    # 时间全局特征
    t_global = len(R)
    if t_global > 0:
        # 电阻全局
        R_x = R.copy()
        R_feats = curve_simple_feats_2(R_x)
        # 功率和能量的全局特征
        P_x = P.copy()
        Q = get_Q_by_P(P_x)  # 能量
        P_x = P.copy()
        P_feats = curve_simple_feats_2(P_x)
        # 电流全局特征
        I_x = I.copy()
        I_feats = curve_simple_feats_2(I_x)
        # 电压全局特征
        U_x = U.copy()
        U_feats = curve_simple_feats_2(U_x)

        feats.append(t_global)  # 全局时间
        feats.extend(R_feats)  # 电阻特征
        feats.extend(P_feats)  # 电功率特征
        feats.append(Q)  # 电热全局特征
        feats.extend(I_feats)  # 电流特征
        feats.extend(U_feats)  # 电压特征
    else:
        feats.extend([0. for _ in range(98)])
    return feats


def get_rui_simple_feats_3(big_sec, R, U, I, P):
    def sort_item_0(item):
        return item[0]

    feats = []
    r_ls = []
    if big_sec[0][1] > 0:
        [x_sec, level] = big_sec
        [x1, x2] = x_sec
        r1 = R[x1]
        r2 = R[x2]
        R_sec = np.array(R[x1:x2 + 1])
        r3 = np.min(R_sec)
        x3 = int(np.where(R_sec == np.min(R_sec))[0][0] + x1)
        r4 = np.max(R_sec)
        x4 = int(np.where(R_sec == np.max(R_sec))[0][0] + x1)
        r_ls.extend([[x1, r1], [x2, r2], [x3, r3], [x4, r4], ])
        r_ls.sort(key=sort_item_0)

        for item in r_ls:
            x, r = item
            feats.append(x)
            feats.append(r)
            feats.append(P[x])
            feats.append(U[x])
            feats.append(I[x])

        len_ = len(r_ls)
        secs = []
        for ii in range(0, len_ - 1):
            secs.append([r_ls[ii][0], r_ls[ii + 1][0]])
        for sec in secs:
            p_sec = P[sec[0]: sec[1]]
            q = get_Q_by_P(p_sec)
            feats.append(q)
            feats.append(q ** (1 / 3))
            feats.append(q ** (4 / 3))

        curve={
            'R':R, 'U':U, 'I':I, 'P':P
        }
        for name in ['I','U','R','P']:
            percenttile = np.percentile(curve[name], [25, 50, 75])  # 1/4,1/2,4/3分位值
            hist, bins = np.histogram(curve[name], 10)  # 分布直方图
            feats.extend([percenttile[i] for i in range(3)])
            feats.extend([hist[i] for i in range(10)])
            # feats.extend([bins[i] for i in range(11)])

    else:
        feats.extend([0. for _ in range(29)])
    return feats


def get_rui_sec_actual_parmas(R, U, I, P):
    sp_i, sp_u, sp_r, sp_p, sp_q, sp_t = 0., 0., 0., 0., 0., 0
    R = np.array(R.copy())
    U = np.array(U.copy())
    I = np.array(I.copy())
    P = np.array(P.copy())
    R = R[np.where(R > 0)]
    U = U[np.where(U > 0)]
    I = I[np.where(I > 0)]
    P = P[np.where(P > 0)]

    sp_t = R.shape[0]
    sp_i = np.mean(I)
    sp_u = np.mean(U)
    sp_r = np.mean(R)
    sp_q = get_Q_by_P(P)
    sp_p = np.mean(P)
    return {
        'sp_i': sp_i,
        'sp_u': sp_u,
        'sp_r': sp_r,
        'sp_q': sp_q,
        'sp_p': sp_p,
        'sp_t': sp_t,
    }
