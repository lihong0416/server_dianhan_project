import datetime
import pandas as pd
import math
import re
from collections import deque
from dateutil.parser import parse


def get_nowdate_str(format_str='%Y-%m-%d %H:%M:%S'):
    return format(datetime.datetime.now(), format_str)


def set_datetime_format_for_dfcol(this_df, col_name, format_str="%Y-%m-%d %H:%M:%S"):
    # 设置统一的时间戳
    this_df[col_name] = pd.to_datetime(this_df[col_name], format=format_str)


def datetime_to_str(dt_obj=None, format='%Y-%m-%d %H:%M:%S'):
    if dt_obj == None or not isinstance(dt_obj, datetime.datetime):
        dt_obj = datetime.datetime.now()
    return '{}'.format(dt_obj.strftime(format))


def make_date_dict(this_str):
    '''
    自动将日期字符串转为日期字典。若日期字符串无法转换，则返回 False。
    Args:
        this_str (str): 日期字符串

    Returns:
        dict or bool: 转换成功返回日期字典，否则返回False。
    '''
    this_dict = {
        'yyyy': '',  # 4位年
        'mm': '',  # 2位月
        'dd': '',  # 2位日
    }
    for this_sep in ['-', '/', '_']:
        ymd_formats = [f'%Y{this_sep}%m{this_sep}%d', f'%d{this_sep}%m{this_sep}%Y', f'%m{this_sep}%d{this_sep}%Y']
        for format_str in ymd_formats:
            try:
                dt_obj = datetime.datetime.strptime(this_str, format_str)
                yyyy = dt_obj.year
                mm = dt_obj.month
                dd = dt_obj.day
                this_dict['yyyy'] = f"{yyyy:4d}"
                this_dict['mm'] = f"{mm:02d}"
                this_dict['dd'] = f"{dd:02d}"
                return this_dict
            except:
                pass
    return False


def str2datetime(datetime_str):
    '''
    自动将字符串转为datetime.datetime对象
    Args:
        datetime_str: 日期字符串

    Returns:
        datetime.datetime or bool, 日期对象，如果转换失败则返回False

    '''
    try:
        dt_obj = parse(datetime_str)
        return dt_obj
    except:
        pass
    assert isinstance(datetime_str,str), "The type of input is not string!"
    sep_list = [' ', '-', '/', '_', ':', '.']
    que_1 = deque()  # 队列
    que_2 = deque()  # 队列
    que_1.append(datetime_str)
    max_len_1st = 0
    max_len_2nd = -1
    while max_len_1st > max_len_2nd:
        while len(que_1) > 0:
            this_str = que_1.popleft()
            is_enque = False
            for sep in sep_list:
                if sep in this_str:
                    que_2.extend(this_str.split(sep))
                    is_enque = True
                    break
            if not is_enque:
                que_2.append(this_str)
        if len(que_2) >= max_len_1st:
            max_len_2nd = max_len_1st
            max_len_1st = len(que_2)

        while len(que_2) > 0:
            this_str = que_2.popleft()
            is_enque = False
            for sep in sep_list:
                if sep in this_str:
                    que_1.extend(this_str.split(sep))
                    is_enque = True
                    break
            if not is_enque:
                que_1.append(this_str)
        if len(que_1) >= max_len_1st:
            max_len_2nd = max_len_1st
            max_len_1st = len(que_1)
    sec_list = list(que_1) if len(que_1) else list(que_2)

    is_date = False
    is_time = False
    date_list = []
    time_list = []
    date_dict = False
    i_year = -1
    len_sec = len(sec_list)
    for i, sec in enumerate(sec_list):
        if len(sec) == 4:
            i_year = i
    if len_sec == 3 and i_year > -1:
        is_date = True
    elif 0 < len_sec < 5 and i_year < 0:
        is_time = True
    elif len_sec > 4 and i_year > -1:
        is_date = True
        is_time = True
    if is_date and is_time:
        if i_year < 3:
            date_list = sec_list[0:3]
            time_list = sec_list[3:]
        elif i_year > len_sec - 4:
            date_list = sec_list[len_sec - 3:]
            time_list = sec_list[0:len_sec - 3]
    elif is_date and not is_time:
        date_list = sec_list[0:3]
    elif not is_date and is_time:
        time_list = sec_list[0:]
    if len(date_list) > 0:
        date_str = "-".join(date_list)
        date_dict = make_date_dict(date_str)

    time_dict = {
        'hh': '00',  # 2位时
        'mm': '00',  # 2位分
        'ss': '00',  # 2位秒
        'tail': '000',  # 毫秒位，长度不定
        'is': False
    }
    if len(time_list) > 0:
        if len(time_list) > 0 and len(time_list[0]) > 0 and time_list[0].isdigit():
            time_dict['hh'] = time_list[0]
            time_dict['is'] = True
        if len(time_list) > 1 and len(time_list[1]) > 0 and time_list[1].isdigit():
            time_dict['mm'] = time_list[1]
            time_dict['is'] = True
        if len(time_list) > 2 and len(time_list[2]) > 0 and time_list[2].isdigit():
            time_dict['ss'] = time_list[2]
            time_dict['is'] = True
        if len(time_list) > 3 and len(time_list[3]) > 0 and time_list[3].isdigit():
            time_dict['tail'] = time_list[3]
            time_dict['is'] = True
    new_dt_str = f""
    format_str = f""
    if date_dict:
        format_str += "%Y-%m-%d"
        new_dt_str += f"{date_dict['yyyy']}-{date_dict['mm']}-{date_dict['dd']}"
    if time_dict['is']:
        format_str += " %H:%M:%S.%f"
        new_dt_str += f" {time_dict['hh']}:{time_dict['mm']}:{time_dict['ss']}.{time_dict['tail']}"
    if date_dict or time_dict['is']:
        dt_obj = datetime.datetime.strptime(new_dt_str, format_str)
        return dt_obj
    else:
        return False

def str_to_datetime(dt_str,
                    format_str=['%Y-%m-%d %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S.%f', '%Y-%m-%d-%H-%M', '%Y-%m-%d-%H-%M-%S']):
    '''
    将日期字符串转为datetime.datetime对象。
    如果无法转换，则返回False。
    Args:
        dt_str (str): 日期字符串
        format_str (str, (list, tuple)): 日期格式，

    Returns:
        dateTime or bool: 转换的datetime.datetime对象，或者返回False表示转换失败

    '''
    try:
        dt_obj = parse(dt_str)
        return dt_obj
    except:
        pass
    dt_obj = dt_str

    if dt_str == None or dt_str == '':
        return False

    if isinstance(dt_str, float):
        if math.isnan(dt_str):
            return False
    if isinstance(dt_str, str):
        try:  # 传入的格式字符串如果有效则执行，如果报错则执行自动识别字符串
            if isinstance(format_str, str):
                dt_obj = datetime.datetime.strptime(dt_str, format_str)
            if isinstance(format_str, (list, tuple)):
                for format_str in format_str:
                    try:
                        dt_obj = datetime.datetime.strptime(dt_str, format_str)
                        break
                    except:
                        pass
        except:  # 自动识别格式字符串
            # 分秒用空格分隔，若用其他分隔则无法识别
            ymd_sym = '-'  # 默认的日期分隔符
            hms_sym = ':'  # 默认的时间分隔符
            dt_str = dt_str.strip()
            find_spc = re.findall(' ', dt_str)
            if len(find_spc) < 1:
                is_hms = False
            else:
                is_hms = True
            if is_hms:
                str_ls = dt_str.split(' ')
                ymd = str_ls[0]
                hms = str_ls[-1]  # 防止中间出现多个空格
                for sym in [':', '-']:  # 重新确定时间分隔符
                    if sym in hms:
                        hms_sym = sym
                        break
            else:
                ymd = dt_str

            for sym in ['-', '/']:  # 重新确定日期分隔符
                if sym in ymd:
                    ymd_sym = sym
                    break

            is_check_ymd = False
            ymd_format_str = f'%Y{ymd_sym}%m{ymd_sym}%d'
            ymd_formats = [f'%Y{ymd_sym}%m{ymd_sym}%d', f'%d{ymd_sym}%m{ymd_sym}%Y', f'%m{ymd_sym}%d{ymd_sym}%Y']
            for format_str in ymd_formats:
                try:
                    dt_obj = datetime.datetime.strptime(ymd, format_str)
                    ymd_format_str = format_str
                    is_check_ymd = True
                    break
                except:
                    pass
            if is_check_ymd:
                if is_hms:
                    tail = ''
                    if '.' in hms:
                        tail = '.%f'
                    hms_format_st = f'%H:%M:%S{tail}'

                if is_hms:
                    format_str = f"{ymd_format_str} {hms_format_st}"
                else:
                    format_str = f"{ymd_format_str}"
                try:
                    dt_obj = datetime.datetime.strptime(dt_str, format_str)
                except:
                    return False
            else:
                return False

            # 年月日与时
    return dt_obj
