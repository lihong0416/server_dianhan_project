from configparser import ConfigParser

def ini2dict(ini_path):
    '''
    将ini配置文件内容转为python字典。
    字典第一层为section，第二层为option。
    例如：
    ini配置内容：
    [path]
    set = D:\a.txt

    [value]
    x = 2
    y = 10

    转为字典后，
    ini_dict[path][set]的值为"D:\a.txt"；ini_dict[value][x]的值为"2"；ini_dict[value][y]的值为"10"。

    :param ini_path: String。ini文件的路径。
    :return: Dictionary。ini文件转换后的字典。
    '''
    conf = ConfigParser()
    conf.read(ini_path)
    sections_ls = conf.sections()

    result={}

    for section in sections_ls:
        if section not in result.keys():
            result[section]={}
        options = conf.options(section)
        for option in options:
            result[section][option]=conf.get(section, option)

    return result

if __name__ == "__main__":
    result=ini2dict(r'E:\PyProject\xx_yj_sxd_ocr\Settings\Settings.ini')
