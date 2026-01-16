from .data_utils import xml2dict
import numpy as np
import os.path
# 字典中提取曲线，字典的键与Mongo数据库的键基本一致。
# 逻辑是从Mongo数据库中提取数据并转为字典，然后将字典交给该方案。
def get_rui_from_dictAEQ(ds_dict,data_model=None):
    this_dict = ds_dict.copy()
    '''
    RUI曲线换算公式
    每格 I=IMAX*0.1 A = IMAX*1e-4 kA
    每格 U=UMAX*0.01 V = UMAX*1e-2 V
    '''
    IMAX = float(this_dict['imax']) * 1e-4
    UMAX = float(this_dict['umax']) * 1e-2

    u_curve = this_dict['u_curve']
    i_curve = this_dict['i_curve']
    u_list = []
    i_list = []

    if data_model=='file':
        u_curve = u_curve[1:-1]
        i_curve = i_curve[1:-1]
        u_list_str = u_curve.split(',')
        i_list_str = i_curve.split(',')
    elif data_model=='ck':
        u_list_str = u_curve
        i_list_str = i_curve
    else:
        u_list_str = [1]*1000
        i_list_str = [1]*1000
    for kk in u_list_str:
        u_list.append(float(kk))
    for kkk in i_list_str:
        i_list.append(float(kkk))
    u_c = [i * UMAX  for i in u_list][:1000]
    i_c = [i * IMAX  for i in i_list][:1000]
    u_c = [round(num, 2) for num in u_c]
    i_c = [round(num, 2) for num in i_c]
    r_c = [a / b if b != 0 else 0 for a, b in zip(u_c, i_c)]
    r_c = [round(num, 4) for num in r_c]
    result = {
        'U': [],
        'I': [],
        'R': [],
        'P': [],
        'non_1': [],
        'non_2': [],
    }

    result['U']=u_c
    result['I']=i_c
    # 电压单位是V,电流单位是kA
    # 直接计算 电阻单位是 微欧
    # 直接计算 电功率单位是 kW

    U = np.array(result['U'])
    I = np.array(result['I'])
    R = np.divide(U, I, out=np.zeros_like(U), where=I != 0)

    # max_R = np.max(R)
    diff = np.diff(R)
    max_R_loc = np.where(np.abs(diff) > 0.07)
    max_R_loc_ls = max_R_loc[0].tolist()
    for dloc in max_R_loc_ls:
        for i_c, loc in enumerate([dloc, dloc + 1]):
            if R[loc] < 0.0001:
                R[loc] = 0.
                continue
            l_border = loc - 3 if loc - 3 > 0 else 0
            r_border = loc + 4 if loc + 4 < U.shape[0] else U.shape[0]
            rl1 = loc - l_border if i_c == 0 else loc - l_border - 1
            rl2 = loc - l_border + 1 if i_c == 0 else loc - l_border
            relative_loc=[rl1,rl2]
            this_relative_loc = loc - l_border
            window_i = I[l_border:r_border]
            window_u = U[l_border:r_border]
            i_valid_loc = np.where(window_i > 0.0001)[0].tolist()
            u_valid_loc = np.where(window_u > 0.0001)[0].tolist()
            relative_valid_loc = [v for v in u_valid_loc if v in i_valid_loc]

            for this_loc in relative_loc:
                if this_loc in relative_valid_loc:
                    relative_valid_loc.remove(this_loc)
            if len(relative_valid_loc) > 1:
                valid_loc = [v + l_border for v in relative_valid_loc]

                r_y = R[valid_loc]
                coef = np.polyfit(x=relative_valid_loc, y=r_y, deg=1)
                new_r = np.polyval(coef, this_relative_loc)
                R[loc] = new_r
    P = U * I
    result['R'] = R.tolist()
    result['P'] = P.tolist()
    return result

