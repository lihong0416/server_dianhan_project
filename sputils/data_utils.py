import os
import re
import json
import math
import time
import os.path
import datetime
import traceback
import xmltodict
import pandas as pd
import numpy as np

from .datetime_utils import get_nowdate_str, str2datetime


def fast_read_xlsx_csv(file_path, sheet_name=0, header=0, chunksize=None):
    '''
    读取CSV或XLSX文件。对于XLSX文件，sheet_name=None则读取所有的sheet，返回pd.DataFrame列表。

    :param file_path:
    :param sheet_name:
    :param header:
    :return:
    '''
    bsname_out = os.path.basename(file_path)
    mnname_out, extname_out = os.path.splitext(bsname_out)
    if extname_out.upper() == '.CSV':
        try:
            df_out = pd.read_csv(file_path, encoding='gbk', chunksize=chunksize)
        except:
            df_out = pd.read_csv(file_path, encoding='utf8', chunksize=chunksize)

    elif extname_out.upper() == '.XLSX' or extname_out.upper() == '.XLS':
        try:
            df_out = pd.read_excel(io=file_path, sheet_name=sheet_name, header=header)
        except:
            df_out = pd.read_excel(io=file_path, sheet_name=sheet_name, engine='openpyxl', header=header)
    else:
        return
    return df_out


def save_df_to_tabel(this_df, save_path):
    '''
    将pd.DataFrame保存为CSV或XLSX文件，文件类型由文件扩展名决定。保存编码为utf-8。
    为了解决CSV文件中文乱码问题，编码为utf-8-sig。

    :param this_df:
    :param save_path:
    :return:
    '''

    dirname = os.path.dirname(save_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    _, extname = os.path.splitext(save_path)
    if extname.upper() == '.XLSX':
        xlsx_writer = pd.ExcelWriter(save_path)
        this_df.to_excel(xlsx_writer, encoding='utf-8', sheet_name='Sheet1', index=False)
        xlsx_writer.save()
    elif extname.upper() == '.CSV':
        this_df.to_csv(save_path, encoding='utf-8-sig', index=False)


def get_row_i_by_key(this_df, key_list, col_i):
    '''
    根据关键字集合和列索引确定行索引号。

    :param key_list:
    :param col_i:
    :return:
    '''
    is_contain = False
    temp_ser_str = this_df.iloc[:, col_i]
    valid_key = None
    row_i = None
    for key in key_list:  # 遍历所有关键字
        bool_ser = temp_ser_str.str.contains(key)  # 对series中每个元素是否包含关键字进行判断
        get_true_ls = list(bool_ser[bool_ser == True].index)  # 获取包含关键字的位置列表
        if len(get_true_ls) > 0:
            valid_key = key
            row_i = get_true_ls[0]
            is_contain = True
            break
    return is_contain, (row_i, col_i), valid_key


def get_col_i_by_key(this_df, key_list, row_i):
    '''
    根据关键字集合和行索引确定列索引号。
    如果关键字在row_i行内出现重复，则返回列索引最小值。

    :param key_list:
    :param row_i: int. 行索引号，从0开始。
    :return:
    '''
    is_contain = False
    temp_ser_str = this_df.iloc[row_i, :]
    valid_key = None
    col_i = None
    for key in key_list:  # 遍历所有关键字
        bool_ser = temp_ser_str.str.contains(key)  # 对series中每个元素是否包含关键字进行判断
        get_true_ls = list(bool_ser[bool_ser == True].index)  # 获取包含关键字的位置列表
        if len(get_true_ls) > 0:
            valid_key = key
            col_i = get_true_ls[0]
            is_contain = True
            break
    return is_contain, (row_i, col_i), valid_key


def nan_count_around(this_df, row_i, col_i, thrd=5):
    '''
    以某单元格为起点，向上下左右寻找nan的数量

    :param this_df:
    :param row_i:
    :param col_i:
    :param thrd:
    :return:
    '''
    num_rows, num_clos = this_df.shape
    up = 0
    down = 0
    left = 0
    right = 0
    if row_i >= 0 and row_i < num_rows and col_i >= 0 and col_i < num_clos:
        # 向上
        now_row_i = row_i
        while now_row_i >= 0 and up <= thrd:
            if pd.isnull(this_df.iloc[now_row_i, col_i]):
                up += 1
                now_row_i -= 1
            else:
                break

        # 向下
        now_row_i = row_i
        while now_row_i < num_rows and down <= thrd:
            if pd.isnull(this_df.iloc[now_row_i, col_i]):
                down += 1
                now_row_i += 1
            else:
                break
        # 向左
        now_col_i = col_i
        while now_col_i >= 0 and left <= thrd:
            if pd.isnull(this_df.iloc[row_i, now_col_i]):
                left += 1
                now_col_i -= 1
            else:
                break

        # 向右
        now_col_i = col_i
        while now_col_i < num_clos and right <= thrd:
            if pd.isnull(this_df.iloc[row_i, now_col_i]):
                right += 1
                now_col_i += 1
            else:
                break
    return {'up': up, 'down': down, 'left': left, 'right': right}


def sort_weight(item):
    return item[2]


def square_search(row_i, col_i, num_row, num_cols, area=5):
    coord_ls = []
    min_row = row_i - area if row_i - area >= 0 else 0
    max_row = row_i + area if row_i - area < num_row else num_row - 1
    min_col = col_i - area if col_i - area >= 0 else 0
    max_col = col_i + area if col_i + area < num_cols else num_cols - 1
    for this_c_i in range(min_col, max_col + 1, 1):
        for this_r_i in range(min_row, max_row + 1, 1):
            weight = 0
            if this_c_i == col_i:
                weight = 0 + abs(this_c_i - col_i) + abs(this_r_i - row_i)
            elif this_r_i == row_i:
                weight = 50 + abs(this_c_i - col_i) + abs(this_r_i - row_i)
            else:
                weight = 100 + abs(this_c_i - col_i) + abs(this_r_i - row_i)

            coord_ls.append([this_r_i, this_c_i, weight])
    coord_ls.sort(key=sort_weight)
    return coord_ls


def get_range_index(now, range_max, width=10):
    def sort_weight(item):
        return item[1]

    result = []
    min = now - width if now - width >= 0 else 0
    max = now + width if now + width < range_max else range_max - 1

    for this_i in range(min, max + 1, 1):
        weight = abs(this_i - now)

        result.append([this_i, weight])
    result.sort(key=sort_weight)
    return result


def is_null_for_pd(x):
    return pd.isnull(x) or (x == '') or (not x)


def is_not_null_for_pd(x):
    return (not pd.isnull(x)) and (not x == '') and (x)


def make_save_path(save_dir, kernel_name, extname='.csv', sep='#', version=1):
    save_name = f"{kernel_name}{sep}v{get_nowdate_str(format_str='%Y%m%d')}-{version}{extname}"
    return os.path.join(save_dir, save_name)


def convert_excel_i_to_matrix_i(excel_index):
    return excel_index - 1


def excel_convert_letter_to_number(letter, columnA=0):
    """
    字母列号转数字
    columnA: 你希望A列是第几列(0 or 1)? 默认0
    return: int
    """
    ab = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter0 = letter.upper()
    w = 0
    for _ in letter0:
        w *= 26
        w += ab.find(_)
    return w - 1 + columnA


def excel_convert_number_to_letter(number, columnA=0):
    """
    数字转字母列号
    columnA: 你希望A列是第几列(0 or 1)? 默认0
    return: str in upper case
    """
    ab = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n = number - columnA
    x = n % 26
    if n >= 26:
        n = int(n / 26)
        return excel_convert_number_to_letter(n, 1) + ab[x + 1]
    else:
        return ab[x + 1]


def template_extract(template_path):
    if isinstance(template_path, str):
        header_df = fast_read_xlsx_csv(template_path, sheet_name=0)
    elif isinstance(template_path, pd.DataFrame):
        header_df = template_path
    header_df = header_df.fillna('')
    header_ls = header_df.to_dict("record")  # 按行将pd转dict
    return header_df, header_ls


def get_pd_dtype(type_str):
    type_dict = {
        'object': ['str', 'string'],
        'int64': ['int'],
        'float64': ['float'],
        'bool': ['bool'],
        'datetime64': ['datetime'],
    }
    for key in type_dict.keys():
        if type_str in type_dict[key]:
            return key
    return 'object'


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def remove_err(input, thr=5):
    if len(input) > 0:
        remove_ls = []
        len_ = len(input)
        pt = len_ - 1
        while pt > 0:
            if abs(input[pt] - input[pt - 1] > thr):
                for i in range(pt, len_, 1):
                    remove_ls.append(input[i])
            pt -= 1
        # result = list(set(input).difference(set(remove_ls)))
        # !!! 非常隐秘的Bug!!! 序列首数是395时，尾数在513~701时，set集合的首数由395变为512
        # 解决方法：先将List中的int转为string，然后List转为set进行集合运算
        str1_ls = [str(i) for i in input]
        str2_ls = [str(i) for i in remove_ls]
        set1 = set(str1_ls)
        set2 = set(str2_ls)
        set_diff = set1.difference(tuple(set2))
        result = list(map(int, set_diff))
        result.sort()
        return result
    else:
        return input


def extract_info(data_df, header_list, header_en=True):
    num_row, num_cols = data_df.shape[0], data_df.shape[1]
    result_dict = {}
    # 重制属性header参数
    headers_ls = []
    if header_en:
        feat_name = 'feat_en_name'
    else:
        feat_name = 'feat_ch_name'
    for header in header_list:
        # is_extract项大于1，使用default_value作为该项值；等于1，则在data_df中提取数据
        result_dict[header[feat_name]] = np.nan

        if int(header['is_extract']) > 1:
            result_dict[header[feat_name]] = header['default_value']
        elif int(header['is_extract']) == 1:
            # 提取该项表头主要信息
            if int(header['num_title_level']) == 1:
                col_name = str(header['level_1']).strip()
            elif int(header['num_title_level']) == 2:
                col_name = str(header['level_2']).strip()
            header_row_i = convert_excel_i_to_matrix_i(int(header['row_i']))
            header_col_i = excel_convert_letter_to_number(str(header['col_i']), columnA=0)
            data_direction = str(header['data_direction'])
            num_data = int(header['num_data'])
            data_offset = int(header['data_offset'])
            col_dtype = get_pd_dtype(header['data_type'])
            not_null = int(header['not_null']) if is_number(str(header['not_null'])) else 0

            # 在df中按模板中的定义取关键字并进行验证
            is_contain = False
            if header_row_i < num_row and header_col_i < num_cols:
                check_item = data_df.iloc[header_row_i, header_col_i]
                # print('col_name: {}, check_item: {}'.format(col_name, check_item))
                if is_not_null_for_pd(check_item):
                    check_item = str(check_item)
                    if col_name in check_item:
                        is_contain = True

            # 如果上一步验证失败，则在附近行继续搜索。按行搜索
            if not is_contain:
                rows_ls = get_range_index(header_row_i, range_max=num_row, width=20)
                for this_row_i, _ in rows_ls:
                    is_contain, (header_row_i, header_col_i), valid_key = get_col_i_by_key(data_df, key_list=[col_name],
                                                                                           row_i=this_row_i)
                    if is_contain:
                        break
            if is_contain:
                if data_direction.upper() == 'DOWN':
                    first_row_i = header_row_i + data_offset
                    first_col_i = header_col_i
                elif data_direction.upper() == 'RIGHT':
                    first_row_i = header_row_i
                    first_col_i = header_col_i + data_offset
                header_item = {}
                header_item['col_name'] = header[feat_name]
                header_item['header_row_i'] = header_row_i
                header_item['header_col_i'] = header_col_i
                header_item['first_row_i'] = first_row_i
                header_item['first_col_i'] = first_col_i
                header_item['num_data'] = num_data
                header_item['col_dtype'] = col_dtype
                header_item['not_null'] = not_null
                header_item['data_direction'] = data_direction
                headers_ls.append(header_item)

    # 根据非空列筛选行号
    not_null_cols = {}
    is_screen = False
    for header in headers_ls:
        if header['not_null'] == 1 and int(header['num_data']) > 1:
            is_screen = True
            not_null_cols[header['col_name']] = (
                data_df.iloc[header['first_row_i']:, header['first_col_i']].apply(is_not_null_for_pd))
    if is_screen:
        temp_pd = pd.DataFrame(not_null_cols)
        bool_pd = temp_pd.all(axis=1)
        have_data_i_ls = bool_pd[bool_pd == True].index.tolist()
        have_data_i_ls = remove_err(have_data_i_ls)

    # 提取数据
    for header in headers_ls:
        result_dict[header['col_name']] = np.nan
        # print(header['col_name'])
        if int(header['num_data']) == 1:
            result_dict[header['col_name']] = data_df.iloc[header['first_row_i'], header['first_col_i']]
            if is_null_for_pd(result_dict[header['col_name']]):
                if header['col_dtype'] in ['int64', 'float64', 'bool']:
                    result_dict[header['col_name']] = result_dict[header['col_name']].fillna(0)
                else:
                    result_dict[header['col_name']] = str(result_dict[header['col_name']])
                    result_dict[header['col_name']] = ''

            if isinstance(result_dict[header['col_name']], str):
                result_dict[header['col_name']] = result_dict[header['col_name']].strip()

        else:
            if header['data_direction'].upper() == 'DOWN':
                result_dict[header['col_name']] = data_df.iloc[have_data_i_ls, header['first_col_i']]
                if result_dict[header['col_name']].isnull().any(axis=0):
                    if header['col_dtype'] in ['int64', 'float64', 'bool']:
                        result_dict[header['col_name']] = result_dict[header['col_name']].fillna(0)
                    else:
                        result_dict[header['col_name']] = result_dict[header['col_name']].fillna('')
                result_dict[header['col_name']] = result_dict[header['col_name']].astype(header['col_dtype'])
                if result_dict[header['col_name']].dtype == 'object':
                    result_dict[header['col_name']] = result_dict[header['col_name']].fillna('')
                    # print(header['col_name'])
                    # print(result_dict[header['col_name']].dtype)
                    result_dict[header['col_name']] = result_dict[header['col_name']].astype(str)
                    result_dict[header['col_name']] = result_dict[header['col_name']].str.strip()
    result_df = pd.DataFrame(result_dict)
    result_df.index = range(1, len(result_df) + 1)
    result_df = result_df.fillna('')
    return result_df


def xml2dict(xml_path):
    with open(xml_path, 'r', encoding='utf8') as fp:
        xml_data = fp.read()
    dict_data = xmltodict.parse(xml_data)
    return dict_data


def dict_save_to_json(save_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(save_dict, f)


def read_json_to_dict(json_path):
    return json.load(open(json_path, 'r'))


# 按照输入列表在一条键值对数据（字典或pandas df)中取值,
def convert_basic_type(data, new_type):
    # data 类型与 new_type相同，无需转换
    if isinstance(data, new_type):
        return data
    # new_type是数值型
    if new_type in (int, float):
        if is_number(data):  # 如果new_type是int, float等数值型，先判断data是否能转为数值
            try:
                return new_type(float(data))
            except Exception as e:
                raise Exception(traceback.format_exc())
        else:  # data不能转为数值，则
            raise Exception(f'Type of Input {data} cannot be converted to numerical type!')
    # new_type是日期类型
    if new_type == datetime.datetime:
        if isinstance(data, datetime.datetime):  # data类型是datetime.datetime，无需转换
            return data
        if isinstance(data, time.struct_time):  # data类型是time.struct_time
            data = time.strftime('%Y-%m-%d %H:%M:%S', data)
            return str2datetime(data)
        if isinstance(data, str):  # data类型是字符串
            newdata = str2datetime(data)
            if newdata:  # 转换成功
                return newdata
            else:  # 转换失败，返回原数据
                raise Exception(f'Type of Input {data} cannot be converted to datetime.datetime type!')
        if isinstance(data, (float, int)):  # 认为类型是时间戳
            return datetime.datetime.fromtimestamp(data)
        raise Exception(f'Type of Input {data} cannot be converted to datetime.datetime type!')
    # data是nan,返回np.nan
    if isinstance(data, float) and math.isnan(data):
        return np.nan
    # new_type是其他类型
    try:
        return new_type(data)
    except:
        raise Exception(traceback.format_exc())  # 把报错信息raise出去


# 按照输入列表在一条键值对数据（字典或pandas df)中取值,
def extract_data_from_kv(item, input_keys=None, dtype_dict=None, default_type=None):
    item = item.copy()
    this_dict = {}
    # 将pd.Series转为dict
    if isinstance(item, pd.Series):
        item = item.to_dict()
    # 设置遍历的key集合，如果input_keys已设置且大于1，则使用input_keys；否则，使用数据item的key集合
    if input_keys != None and isinstance(input_keys, (list, tuple, set)) and len(input_keys) > 0:  # 设置了input_keys
        keys = input_keys.copy()
    else:
        keys = item.keys()
    for key in keys:  # 遍历input_keys的所有key
        if dtype_dict != None:  # 已设置字段类型字典dtype_dict
            if isinstance(dtype_dict, type):  # 如果dtype_dict是数据类型
                this_dict[key] = convert_basic_type(item[key], dtype_dict)
                continue
            if isinstance(dtype_dict, dict):  # 如果dtype_dict是字典
                if key in dtype_dict:  # 数据的key在dtype_dict字典中已经设置
                    this_dict[key] = convert_basic_type(item[key], dtype_dict[key])
                else:  # 数据的key在dtype_dict字典中未设置，使用默认类型进行转换
                    if default_type != None and isinstance(default_type, type):
                        this_dict[key] = convert_basic_type(item[key], default_type)
                    else:
                        this_dict[key] = item[key]
        else:  # 未设置字段类型字典dtype_dict，则用默认类型进行转换
            if default_type != None and isinstance(default_type, type):
                this_dict[key] = convert_basic_type(item[key], default_type)
            else:
                this_dict[key] = item[key]
    return this_dict


def save_onedata_to_csv(df, save_path):
    if not os.path.isfile(save_path):
        df.to_csv(save_path, mode='a', header=True, index=False)
    else:
        df.to_csv(save_path, mode='a', header=False, index=False)


def check_keys(keys, kv_data):
    if not isinstance(keys, (list, tuple)):
        raise Exception('The type of kernels is not in (list, tuple).')
    if not isinstance(kv_data, (dict)):
        raise Exception('The type of kv_data is not dict.')
    for kernel_name in keys:
        if kernel_name not in kv_data.keys():
            return False, kernel_name
        if isinstance(kv_data[kernel_name], float) and np.isnan(kv_data[kernel_name]):
            return False, kernel_name
    return True, ''


def make_spot_tag(kv_data):
    if 'spot_tag' in kv_data.keys():
        return kv_data['spot_tag']
    elif 'spotTag' in kv_data.keys():
        return kv_data['spotTag']
    else:
        return f"{kv_data['timerName']}_{kv_data['progNo']}"


def search_robot(timername):
    pstr = r"R\d+"
    pattern = re.compile(pstr)
    reobj = pattern.search(timername)
    start, end = reobj.span()
    return timername[start:end], start, end


def process_timername(timername, units):
    assert isinstance(timername, str), "The type of input is not string!"
    _, start, end = search_robot(timername)
    new_start = start - units if start - units >= 0 else 0
    timer_robot = timername[new_start:end]
    return timer_robot
