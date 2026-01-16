import time

import numpy as np
from scipy import stats, signal
from scipy.optimize import leastsq
import math


def standardization(data, mean, std):
    return (data - mean) / std


def rescale_min_max(data, min, max):
    return (data - min) / (max - min)


def mean_std(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return mu, sigma


def max_min(data):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    return min, max


def cal_norm_interval(alpha, mean, std):
    return stats.norm.interval(alpha, loc=mean, scale=std)


def curve_diff(y):
    if not isinstance(y, (np.ndarray, list)):
        raise Exception('The type of y is not in (np.ndarray, list).')
    diff = np.diff(y)  # 一阶导数
    quad = np.diff(diff)  # 二阶导数
    # 为了补np.diff缺失的1位，用0插入至第1位.一阶导数插1个0，二阶导数插2个0
    diff2 = np.r_[np.array([0]), diff]
    quad2 = np.r_[np.array([0, 0]), quad]
    return diff2, quad2


def curve_diff_1(y):
    if not isinstance(y, (np.ndarray, list)):
        raise Exception('The type of y is not in (np.ndarray, list).')
    diff = np.diff(y)  # 一阶导数
    # 为了补np.diff缺失的1位，用0插入至第1位.一阶导数插1个0
    diff2 = np.r_[np.array([0]), diff]
    return diff2


def moving_conv_smooth(y, windowsize):
    '''
    滑动平均滤波法进行曲线平滑
    :param interval:
    :param windowsize:
    :return:
    '''
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(y, window, 'same')
    return re


def bsn_moving_mean_smooth(y, windowsize):
    '''
    滑动平均滤波法进行曲线平滑
    '''
    if isinstance(y, (list, tuple)):
        y = np.array(y)
    if not isinstance(y, np.ndarray):
        raise Exception('The type of y is not in [list, tuple, np.ndarray].')

    half_win = int(windowsize / 2)
    shape = y.shape[0]
    result = np.zeros_like(y)
    for ii in range(shape):
        left = ii - half_win if ii - half_win > 0 else 0
        right = ii + half_win + 1 if ii + half_win + 1 < shape else shape
        win_arr = y[left:right]
        val = np.mean(win_arr)
        result[ii] = val
    return result


# 根据x和y两个集合拟合直线
def linefit(x, y):
    # 返回结果即每条直线，信息包括：is_ok，k, b
    # is_ok=False表示直线为竖直线，例如：x=5
    # is_ok=True, k表示直线斜率；否则，k表示竖直线的横坐标，即x=k
    # is_ok=True, b表示直线截距；否则，b=0
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    # 运算
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]

    # # 如果abs(sx * sx / N - sxx)接近0，则说明直线近似为垂直
    # if abs(sx * sx / N - sxx) < 30:
    #     return False, sx / N, 0

    # 计算k和b
    k = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - k * sx) / N

    return k, b


def bsn_rui_least_square_smooth(y, windowsize=10, degree=1, small_win_count=20, outlier_thr=0.2, zero_thr=0.001,
                                is_relative=True, stride=1,strid_win=6):
    '''
    波士内rui曲线自适应窗口分段最小二乘拟合
    '''
    if isinstance(y, (list, set, tuple)):
        y = np.array(list(y))
    if not isinstance(y, np.ndarray):
        raise Exception('The type of y is not np.ndarray')

    # 寻找导数异常点，这些异常点对应曲线的骤降点
    if is_relative:
        y_n = y / np.max(abs(y))  # 如果使用相对值判断异常点，先取绝对值再进行归一化
    else:
        y_n = y  # 不进行归一化，仅取绝对值
    diff_1 = curve_diff_1(y_n)  # 求1阶导数
    abs_diff = abs(diff_1)  # 1阶导数绝对值
    outliers = np.where(abs_diff > outlier_thr)  # 根据1阶导数曲线判断异常点，该变量是数组

    # 求非0值的中位数并作为阈值，用于测量某段是否为0值段
    non_zero_y = y[np.where(y != 0)]  # 取曲线的非0值
    y_median = np.median(abs(non_zero_y))  # 曲线非0值绝对值的中位数

    # 创建保存结果的数组，即平滑后的曲线
    result = np.zeros_like(y)

    # 生成异常点队列
    if len(outliers) > 0:
        outliers = outliers[0].tolist()
    else:
        outliers = []

    # 为异常点队列补曲线的首位置(0)和尾位置(len_curve-1)
    if 0 not in outliers:
        outliers.append(0)
    if y.shape[0] - 1 not in outliers:
        outliers.append(y.shape[0] - 1)
    outliers.sort()

    # 对又异常点之间形成的各段进行平滑处理
    for ii in range(len(outliers) - 1):
        start, end = outliers[ii], outliers[ii + 1]  # 异常点作为段的起点与终点
        # 异常点的值直接作为新曲线的值，保持异常点的准确性
        result[start] = y[start]
        result[end] = y[end]

        if end - (start + 1) > 1:  # 如果段长度大于1，则进行最小二乘平滑处理
            if end - (start + 1) <= windowsize:  # 自适应窗口大小
                small_windows = [int(windowsize * (small_win_count - prop) / small_win_count) for prop in
                                 range(1, small_win_count, 1)]
                if 0 in small_windows:
                    small_windows.remove(0)
                for small_win in small_windows:
                    if end - (start + 1) <= small_win:
                        windowsize = small_win
            sec_arr = y[start + 1:end]

            sec_mean = np.mean(abs(sec_arr))
            if sec_mean < y_median * zero_thr:  # 如果是0值段，则跳过
                continue

            # 在段内使用滑窗
            half_win = int(windowsize / 2)
            stride_half_win = int(strid_win / 2)
            shape = sec_arr.shape[0]

            for jj in range(0,shape,stride):
                left = jj - half_win if jj - half_win > 0 else 0
                right = jj + half_win if jj + half_win < shape - 1 else shape - 1
                center = jj - left
                win_arr = y[start + 1 + left:start + 1 + right + 1]
                win_x = np.arange(0, win_arr.shape[0])
                coef = np.polyfit(win_x, win_arr, degree)
                val = np.polyval(coef, center)
                result[start + 1 + jj] = val
            for jj in range(0, shape, stride):
                if jj < shape - 1 and stride>1:
                    for mm in range(stride-1):
                        nn=jj+1+mm
                        left = nn - stride_half_win if nn - stride_half_win > 0 else 0
                        right = nn + stride_half_win if nn + stride_half_win < shape - 1 else shape - 1
                        win_arr = y[start + 1 + left:start + 1 + right + 1]
                        result[start + 1 + nn] = np.mean(win_arr)
        elif end - (start + 1) == 1:  # 如果段长度等于1，则直接将值赋予新曲线
            result[start + 1] = y[start + 1]
    return result


def weight_filter_1d(windowsize, prop=3, random_sort=False, factor=0.1):
    win_one = np.ones(windowsize)
    win_cum = np.cumsum(win_one)
    win_cum = win_cum * prop
    max_val = np.max(win_cum) + prop
    win_total = np.full(win_one.shape, max_val)
    win_filter = (win_total - win_cum) / np.sum(win_cum)
    if random_sort:
        rng = np.random.default_rng()
        win_filter = rng.permutation(win_filter)
    return win_filter


def bsn_weighted_mean_smooth(y, windowsize):
    '''
    波士内带权重的滑动平均滤波法进行曲线平滑
    '''
    if windowsize <= 0:
        raise Exception('windowsize must be greater than 0.')

    if not isinstance(windowsize, (int, float)):
        raise Exception('The type of windowsize is not in (int, float).')

    if not isinstance(windowsize, int):
        windowsize = int(math.ceil(windowsize))

    if isinstance(y, (list, set, tuple)):
        y = np.array(list(y))
    if not isinstance(y, np.ndarray):
        raise Exception('The type of y is not np.ndarray')

    shape = y.shape[0]
    result = np.zeros_like(y)

    win_filter = weight_filter_1d(windowsize)

    for ii in range(shape):
        left = ii
        right = ii + windowsize if ii + windowsize < shape else shape
        win_arr = y[left:right]
        real_width = right - left
        if real_width < windowsize:
            win_filter = weight_filter_1d(real_width)
        val = np.sum(win_arr * win_filter)
        result[ii] = val
    return result


def savgol_smooth(y, windowsize, k):
    '''
    Savitzky-Golay滤波法进行曲线平滑
    '''
    return signal.savgol_filter(y, windowsize, k)


def search_extreme_point(y, mode=1, mode_class=np.greater):
    '''
    求曲线上的极大极小值。在rui曲线上不太好用。
    :param y: 曲线数组，np.Array
    :param mode: int,大于0的数表示取极大值，小于等于0的数表示取极小值
    :param mode_class:
    :return:
    '''
    if isinstance(mode, (int, float)):
        mode_class = np.greater if mode > 0 else np.less
    return signal.argrelextrema(y, mode_class)


class LeastSquareSmooth(object):
    '''
    用最小二乘法进行曲线平滑处理
    '''

    def __init__(self):
        super(LeastSquareSmooth, self).__init__()

    def func(self, x, p):
        f = np.poly1d(p)
        return f(x)

    def residuals(self, p, x, y, reg):
        regularization = 0.1  # 正则化系数lambda
        ret = y - self.func(x, p)
        if reg == 1:
            ret = np.append(ret, np.sqrt(regularization) * p)
        return ret

    def leastsquare_fit(self, data, k=10, order=4, reg=1, is_deriv=False):  # k为求导窗口宽度,order为多项式阶数,reg为是否正则化
        '''
        最小二乘法拟合。
        :param data: 待拟合的数据. list, tuple, np.ndarray
        :param k: 滑动窗口的宽度
        :param order: 多项式阶数
        :param reg: 是否进行正则
        :param is_deriv: 是否对曲线求导
        :return:
        '''
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise Exception('The type of input is not in (list, tuple, np.ndarray).')
        if isinstance(data, (list, tuple)):
            l = len(data)
        if isinstance(data, (np.ndarray)):
            l = data.shape[0]
        fit_vals = []
        der1 = []
        der2 = []
        der3 = []

        step = 2 * k + 1
        p = [1] * order
        for i in range(0, l, step):
            if i + step < l:
                y = data[i:i + step]
                x = np.arange(i, i + step)
            else:
                y = data[i:]
                x = np.arange(i, l)
            try:
                r = leastsq(self.residuals, p, args=(x, y, reg))
            except:
                raise Exception("Least Square Fitting Failed.")
            fun = np.poly1d(r[0])  # 返回拟合方程系数
            fit_v = fun(x)
            fit_vals.extend(fit_v.tolist())

            if is_deriv:
                df_1 = np.poly1d.deriv(fun)  # 求得导函数
                df_2 = np.poly1d.deriv(df_1)
                df_3 = np.poly1d.deriv(df_2)
                df_value = df_1(x)
                der1.extend(df_value.tolist())
                df2_value = df_2(x)
                der2.extend(df2_value.tolist())
                df3_value = df_3(x)
                der3.extend(df3_value.tolist())
        return fit_vals, der1, der2, der3


def check_num_sign(num):
    if not isinstance(num, (int, float)):
        raise Exception('The type of input is not in (int, float)')
    if num > .0:
        return 1
    elif num < .0:
        return -1
    else:
        return 0


def find_coord_continuous_section(coords, is_single=False):
    '''
    输入不完全连续的坐标序列，查找坐标值序列中的连续区间。
    '''
    con_groups = []
    len_groups = []
    len_input = len(coords)
    if len_input == 0:
        return con_groups, len_groups

    if is_single and len_input == 1:
        con_groups.append([coords[0], coords[0]])  # 为了与后面的格式保持一致
        len_groups.append(len_input)
        return con_groups, len_groups

    temp = []
    for idx in range(1, len_input, 1):
        pre_idx = idx - 1

        if idx == 1:
            temp.append(coords[pre_idx])
        if coords[idx] - coords[pre_idx] > 1:
            temp.append(coords[pre_idx])
            if is_single == False:
                if temp[0] < temp[1]:
                    con_groups.append(temp.copy())
                    len_groups.append(temp[1] - temp[0] + 1)
            else:
                con_groups.append(temp.copy())
                len_groups.append(temp[1] - temp[0] + 1)
            temp.clear()
            temp.append(coords[idx])
        if idx == len_input - 1 and len(temp) == 1:
            temp.append(coords[idx])
            if temp[0] < temp[1]:
                con_groups.append(temp.copy())
                len_groups.append(temp[1] - temp[0] + 1)
            temp.clear()
    return con_groups, len_groups


def find_samesign_section(vals):
    '''
    寻找同号数值的连续区域和异号区的交界点。区域分为0、正号、负号三种。
    该方法的使用场景是取得一阶导数的三个区和特殊点
    '''
    if not isinstance(vals, (list, tuple)):
        raise Exception('The type of input is not in (list,tuple).')
    # 初始化结果返回字典
    result = {
        1: {'val': [], 'coord': []},
        -1: {'val': [], 'coord': []},
        0: {'val': [], 'coord': []},
    }
    # 如果输入列表为空
    len_input = len(vals)
    if len_input == 0:
        return result
    # 如果输入列表长度为1
    if len_input == 1:
        sign = check_num_sign(vals[0])
        result[sign]['val'].append([vals[0], vals[0]])  # 与后面的格式保持一致
        result[sign]['coord'].append([0, 0])
        return result

    # 如果输入列表的长度大于1
    temp_coord = []
    temp_vals = []
    for idx in range(1, len_input, 1):
        pre_idx = idx - 1
        this_sign = check_num_sign(vals[pre_idx])  # 判断前1个位置数字的符号
        temp_vals.append(vals[pre_idx])

        if idx == 1:  # 如果循环刚开始，则将第0位数字放入temp队列中。
            temp_coord.append(pre_idx)
        if check_num_sign(vals[idx]) != this_sign:  # 判断当前位置和前一个位置的数字是否同符号，不同则将前一位置数字加入到temp中
            temp_coord.append(pre_idx)
            result[this_sign]['val'].append(temp_vals.copy())  # 将temp队列副本按符号放入到result中
            result[this_sign]['coord'].append(temp_coord.copy())  # 将temp队列副本按符号放入到result中
            temp_vals.clear()
            temp_coord.clear()
            this_sign = check_num_sign(vals[idx])  # 判断当前位置的数字符号
            temp_coord.append(idx)  # 将当前位置数字加入到清空后的temp中，作为下一段区间的起点
        if idx == len_input - 1 and len(temp_coord) == 1:
            temp_coord.append(idx)
            temp_vals.append(vals[idx])
            result[this_sign]['val'].append(temp_vals.copy())
            result[this_sign]['coord'].append(temp_coord.copy())
            temp_coord.clear()
    return result


def deriv_mag_order(deriv, base_level=-7):
    '''
    获得导数第一位所在位数
    '''
    if not isinstance(deriv, float):
        try:
            deriv = float(deriv)
        except:
            raise Exception('The type of input cannot be converted to float')
    base_val = 10 ** base_level
    loop = 12
    max_val = 10 ** (loop + base_level)

    # 判断导数的符号
    if deriv > 0:
        sign = 1
    else:
        sign = -1
    abs_deriv = abs(deriv)
    # 若导数值小于检测最小值，则返回设定的最小值
    if abs_deriv < base_val:
        return dict(
            base_level=base_level,
            level_level=0,
            sign=sign)
    if abs_deriv > max_val:
        return dict(
            base_level=base_level,
            level_level=loop,
            sign=sign)
    for level in range(loop):
        mag_order = base_val * (10 ** level)
        val = (abs_deriv / mag_order) * 10
        if val >= 1. and val < 10.:
            return dict(
                base_level=base_level,
                level_level=level,
                sign=sign)


def find_extreme_val(derivs):
    if not isinstance(derivs, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(derivs, (list, tuple)):
        len_ = len(derivs)
    if isinstance(derivs, np.ndarray):
        len_ = derivs.shape[0]

    less_ls = []
    greater_ls = []

    for idx in range(2, len_ - 1, 1):
        pre_idx = idx - 1
        next_idx = idx + 1
        pre_der = float(derivs[pre_idx])
        der = float(derivs[idx])
        next_der = float(derivs[next_idx])
        if (pre_der > .0 and der < .0) \
                or (pre_der > .0 and der == .0) \
                or (pre_der > .0 and next_der < .0) \
                or (pre_der > .0 and next_der == .0):
            greater_ls.append(idx)
        if (pre_der < .0 and der > .0) \
                or (pre_der < .0 and der == .0) \
                or (pre_der < .0 and next_der > .0) \
                or (pre_der < .0 and next_der == .0):
            less_ls.append(idx)
    r_less_groups = (np.array(less_ls),)
    r_greater_groups = (np.array(greater_ls),)
    return r_greater_groups, r_less_groups


def find_extreme_val_2(curve):
    # 在曲线上取得极大值和极小值
    r_less_groups = signal.argrelextrema(curve, np.less)
    r_greater_groups = signal.argrelextrema(curve, np.greater)
    return r_greater_groups, r_less_groups


def find_firstderiv_sections_base_sec(derivs):
    '''
    依据输入的区间寻找一阶导数上各区段。
    '''

    def find_sec(input, sections):
        if input > 0:
            sign = 1
        else:
            sign = -1
        abs_input = abs(input)
        for key in sections.keys():
            section = sections[key]
            if abs_input >= section[0] and abs_input < section[1]:
                if key == 0:
                    result = key
                else:
                    result = key * sign
                return result

    # 0：平缓阶段，特征点主要在这个阶段。1,2：过度阶段。3：快速变化阶段。飞溅会发生在1或2中。
    # 暂时放弃：input_sections = {0: [0, 1.1e-4], 1: [1.1e-4, 1e-3], 2: [1e-3, 1.5e-3], 3: [1.5e-3, 1e10]}
    # if test == 1:
    #     input_sections = {0: [0, 1e-3], 1: [1e-3, 1.5e-3], 2: [1.5e-3, 1e10]}
    # elif test == 2:
    #     input_sections = {0: [0, 5e-4], 1: [5e-4, 1.5e-3], 2: [1.5e-3, 1e10]}
    # else:
    #     input_sections = {0: [0, 5e-4], 1: [5e-4, 1.5e-3], 2: [1.5e-3, 1e10]}
    input_sections = {0: [0, 5e-4], 1: [5e-4, 1.5e-3], 2: [1.5e-3, 1e10]}
    len_input = 2
    result = {}
    result_ls = []
    if isinstance(derivs, (list, tuple)):
        len_input = len(derivs)
    if isinstance(derivs, np.ndarray):
        len_input = derivs.shape[0]
    temp_coord = []
    for idx in range(2, len_input, 1):
        pre_idx = idx - 1
        pre_deriv = derivs[pre_idx]
        this_deriv = derivs[idx]
        pre_level = find_sec(pre_deriv, input_sections)
        this_level = find_sec(this_deriv, input_sections)

        if idx == 2:
            temp_coord.append(pre_idx)
        #
        if pre_level != this_level:
            temp_coord.append(pre_idx)
            if pre_level not in result.keys():
                result[pre_level] = []
            result[pre_level].append(temp_coord.copy())
            result_ls.append([temp_coord.copy(), pre_level])
            temp_coord.clear()
            temp_coord.append(idx)
        if idx == len_input - 1 and len(temp_coord) == 1:
            temp_coord.append(idx)
            # if idx != temp_coord[0]:
            if pre_level not in result.keys():
                result[pre_level] = []
            result[pre_level].append(temp_coord.copy())
            result_ls.append([temp_coord.copy(), pre_level])
            temp_coord.clear()
    return result, result_ls


# def find_i_sec(derivs):
#     '''
#     依据输入的区间寻找一阶导数上各区段。
#     '''
#
#     def find_sec(input, sections):
#         if input > 0:
#             sign = 1
#         else:
#             sign = -1
#         abs_input = abs(input)
#         for key in sections.keys():
#             section = sections[key]
#             if abs_input >= section[0] and abs_input < section[1]:
#                 if key == 0:
#                     result = key
#                 else:
#                     result = key * sign
#                 return result
#
#     input_sections = {0: [0, 5e-1], 1: [5e-1, 1e10]}
#     len_input = 2
#     result = {}
#     result_ls = []
#     if isinstance(derivs, (list, tuple)):
#         len_input = len(derivs)
#     if isinstance(derivs, np.ndarray):
#         len_input = derivs.shape[0]
#     temp_coord = []
#     for idx in range(2, len_input, 1):
#         pre_idx = idx - 1
#         pre_deriv = derivs[pre_idx]
#         this_deriv = derivs[idx]
#         pre_level = find_sec(pre_deriv, input_sections)
#         this_level = find_sec(this_deriv, input_sections)
#
#         if idx == 2:
#             temp_coord.append(pre_idx)
#         #
#         if pre_level != this_level:
#             temp_coord.append(pre_idx)
#             if pre_level not in result.keys():
#                 result[pre_level] = []
#             result[pre_level].append(temp_coord.copy())
#             result_ls.append([temp_coord.copy(), pre_level])
#             temp_coord.clear()
#             temp_coord.append(idx)
#         if idx == len_input - 1 and len(temp_coord) == 1:
#             temp_coord.append(idx)
#             # if idx != temp_coord[0]:
#             if pre_level not in result.keys():
#                 result[pre_level] = []
#             result[pre_level].append(temp_coord.copy())
#             result_ls.append([temp_coord.copy(), pre_level])
#             temp_coord.clear()
#     return result, result_ls


# def find_secondderiv_zero(derivs):
#     if not isinstance(derivs, (list, tuple, np.ndarray)):
#         raise Exception('The type of input is not in (list, tuple, np.ndarray).')
#     if isinstance(derivs, (list, tuple)):
#         len_ = len(derivs)
#     if isinstance(derivs, np.ndarray):
#         len_ = derivs.shape[0]
#
#     # less_ls = []
#     # greater_ls = []
#     result = []
#
#     for idx in range(2, len_, 1):
#         der = float(derivs[idx])
#         if der == 0.:
#             result.append(idx)
#     return result


def curve_integral(curve):
    if not isinstance(curve, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if not isinstance(curve, np.ndarray):
        curve = np.array(curve)

    return curve.sum()


def curve_simple_feats(curve):
    '''
    获得曲线的简单特征
    :param curve:
    :return:
    '''
    if not isinstance(curve, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(curve, np.ndarray):
        curve = curve.tolist()
    if isinstance(curve, tuple):
        curve = list(curve)
    c = curve.copy()
    len_ = len(c)
    c0 = c[0]
    c1 = c[len_ - 1]
    cmin = min(c)
    cmax = max(c)
    cmean = np.mean(c)
    cmin_id = c.index(min(c))
    cmax_id = c.index(max(c))
    ids = [0, cmin_id, cmax_id, len_ - 1]
    ids.sort()
    len_ls = [ids[1] - ids[0], ids[2] - ids[1], ids[3] - ids[2]]
    val_ls = [c[ids[1]] - c[ids[0]], c[ids[2]] - c[ids[1]], c[ids[3]] - c[ids[2]]]
    result = [c0, c1, cmin, cmax, cmean]
    result.extend(len_ls)
    result.extend(val_ls)
    return result


def curve_simple_feats_2(curve):
    '''
    获得曲线的简单特征,与
    :param curve:
    :return:
    '''
    if not isinstance(curve, (list, tuple, np.ndarray)):
        raise Exception('The type of input is not in (list, tuple, np.ndarray).')
    if isinstance(curve, np.ndarray):
        curve = curve.tolist()
    if isinstance(curve, tuple):
        curve = list(curve)
    c = curve.copy()
    len_ = len(c)
    c0 = c[0]  # 起始值
    c1 = c[len_ - 1]  # 终止值
    cmin = min(c)  # 最小值
    cmax = max(c)  # 最大值
    cmean = np.mean(c)  # 平均值
    # cmedian = np.median(c)  # 中位数
    cmin_id = c.index(min(c))  # 最小值位置
    cmax_id = c.index(max(c))  # 最大值位置

    ids = [0, cmin_id, cmax_id, len_ - 1]  # 几个位置的坐标
    ids.sort()
    len_ls = [ids[1] - ids[0], ids[2] - ids[1], ids[3] - ids[2]]  # 几个位置间的长度
    val_ls = [c[ids[1]] - c[ids[0]], c[ids[2]] - c[ids[1]], c[ids[3]] - c[ids[2]]]
    result = [c0, c1, cmin, cmax, cmean]
    percenttile = np.percentile(c, [25, 50, 75])  # 1/4,1/2,4/3分位值
    hist, bins = np.histogram(c, 10)  # 分布直方图
    result.extend([percenttile[i] for i in range(3)])
    result.extend([hist[i] for i in range(10)])
    # result.extend([bins[i] for i in range(11)])
    result.extend(len_ls)
    result.extend(val_ls)
    return result


# def curve_local_simple_feats(curve):
#     if not isinstance(curve, (list, tuple, np.ndarray)):
#         raise Exception('The type of input is not in (list, tuple, np.ndarray).')
#     if isinstance(curve, np.ndarray):
#         curve = curve.tolist()
#     if isinstance(curve, tuple):
#         curve = list(curve)
#     c = curve.copy()
#     len_ = len(c)
#     c0 = c[0]
#     c1 = c[len_ - 1]
#     result = [c0, c1]
#     return result


def confidence_interval_one_dim(data, sigma=-1, alpha=0.05, side_both=True):
    xb = np.mean(data)
    s = np.std(data, ddof=1)
    if sigma > 0:  # sigma已知，枢轴量服从标准正态分布
        Z = stats.norm(loc=0, scale=1.)
        if side_both:  # 求双侧置信区间
            tmp = sigma / np.sqrt(len(data)) * Z.ppf(1 - alpha / 2)
            return (xb - tmp, xb + tmp)
        else:  # 单侧置信下限或单侧置信上限
            tmp = sigma / np.sqrt(len(data)) * Z.ppf(1 - alpha)
            return {'bottom_limit': xb - tmp, 'top_limit': xb + tmp}
    else:  # sigma未知，枢轴量服从自由度为n-1的t分布
        T = stats.t(df=len(data) - 1)
        if side_both:
            tmp = s / np.sqrt(len(data)) * T.ppf(1 - alpha / 2)
            return (xb - tmp, xb + tmp)
        else:
            tmp = s / np.sqrt(len(data)) * T.ppf(1 - alpha)
            return {'bottom_limit': xb - tmp, 'top_limit': xb + tmp}


def confidence_interval_for_rui(data, sigma=-1, alpha=0.05, side_both=True):
    num_data = data.shape[0]
    xb = np.mean(data, axis=0)
    s = np.std(data, ddof=1, axis=0)
    if sigma > 0:  # sigma已知，枢轴量服从标准正态分布
        Z = stats.norm(loc=0, scale=1.)
        if side_both:  # 求双侧置信区间
            tmp = sigma / np.sqrt(num_data) * Z.ppf(1 - alpha / 2)
        else:  # 单侧置信下限或单侧置信上限
            tmp = sigma / np.sqrt(num_data) * Z.ppf(1 - alpha)
    else:  # sigma未知，枢轴量服从自由度为n-1的t分布
        T = stats.t(df=num_data - 1)
        if side_both:
            tmp = s / np.sqrt(num_data) * T.ppf(1 - alpha / 2)
        else:
            tmp = s / np.sqrt(num_data) * T.ppf(1 - alpha)
    return {'mean': xb, 'bottom_limit': xb - tmp, 'top_limit': xb + tmp, 'diff': tmp}


def rui_anomaly_detection_base_on_iqr(data, multiple=1.5):
    mean_up_down = np.percentile(data, [25, 50, 75], axis=0)
    per25 = mean_up_down[0]
    per50 = mean_up_down[1]
    per75 = mean_up_down[2]
    iqr = per75 - per25
    down_limit = per25 - multiple * iqr
    up_limit = per75 + multiple * iqr
    return {'per50': per50, 'bottom_limit': down_limit, 'top_limit': up_limit, 'iqr': iqr}


def softmax(x):
    x -= np.max(x, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), keepdims=True)

    return x
