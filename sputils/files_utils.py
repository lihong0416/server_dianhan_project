import glob
import zipfile
import time
import shutil


def search_files(root, res_ls, corestr=None, excludestr=None, extname=None, is_and=False):
    '''
    用递归的方式搜索文件路径，并将路径放入列表中。
    可设置简单的筛选条件：
    corestr用于设置必须存在于路径中的字符串。可以用List, Tuple, Set同时设置多个字符串。
    excludestr用于设置必须不在路径中的字符串。可以用List, Tuple, Set同时设置多个字符串。
    extname用于设置文件的扩展名。可以用List, Tuple, Set同时设置多个字符串。
    is_and用于设置上述三个条件是取交集还是并集。True为交集，即三个条件必须同时满足。

    例如，我们希望匹配到”dev“和"release"文件夹下，文件名不包含"other"和”debug“字符串，".pyc"和”.py“两种扩展名的文件。
    可设置：corestr=['dev','release']，excludestr=['other','debug']，excludestr=['.pyc','.py']

    :param root: String。递归搜索文件的根目录
    :param res_ls: List<String>。递归搜索结果列表。
    :param corestr: String/List/Tuple/Set。设置必须存在于路径中的字符串，如果为集合类型，元素必须是字符串。
    :param excludestr: String/List/Tuple/Set。设置必须不在于路径中的字符串，如果为集合类型，元素必须是字符串。
    :param extname: String/List/Tuple/Set。设置文件的扩展名，如果为集合类型，元素必须是字符串。
    :param is_and: Boolean。是否设置为3个条件同时满足。若为False，则为3个条件满足其一即可。
    :return: List。满足条件的文件路径列表，元素类型为String。
    '''

    items = glob.glob(os.path.join(root, '*'))
    for path in items:
        if isinstance(path, str):
            if os.path.isdir(path):
                search_files(path, res_ls, corestr, excludestr, extname, is_and)
            elif os.path.isfile(path):

                # 判断corestr是否在文件路径中
                is_corestr = False
                if isinstance(corestr, (list, tuple, set)):
                    for item in corestr:
                        if isinstance(item, str) and item in path:
                            is_corestr = True
                elif isinstance(corestr, str):
                    if corestr in path:
                        is_corestr = True
                elif corestr == None:
                    is_corestr = True

                # 判断excludestr是否不在文件路径中
                is_excludestr = True
                if isinstance(excludestr, (list, tuple, set)):
                    for item in excludestr:
                        if isinstance(item, str) and item in path:
                            is_excludestr = False
                elif isinstance(excludestr, str):
                    if excludestr in path:
                        is_excludestr = False

                # 判断extname是否为文件扩展名
                is_extname = False
                basename = os.path.basename(path)
                _, this_ext = os.path.splitext(basename)
                if isinstance(extname, (list, tuple, set)):
                    for item in extname:
                        if isinstance(item, str) and this_ext == item:
                            is_extname = True
                elif isinstance(extname, str):
                    if this_ext == extname:
                        is_extname = True
                elif extname == None:
                    is_extname = True

                if is_and:
                    if is_corestr and is_excludestr and is_extname:
                        res_ls.append(path)
                else:
                    if is_corestr or is_excludestr or is_extname:
                        res_ls.append(path)
    return res_ls


def backupFile(file_full_path):
    """
    文件备份,返回备份的文件名称。将文本在同目录备份成zip格式，命名为： 原文件全名-年月日时分-backup.zip。例如 abc.txt，备份为： abc.txt-202103141000-backup.zip
    :param file_full_path: 目标文本的绝对路径
    :return: 返回备份的文件名称
    """
    assert os.path.isfile(file_full_path), "zip备份的目标不是文件"
    file_name = os.path.basename(file_full_path)
    file_path = os.path.dirname(file_full_path)
    print("开始备份文件-{}".format(file_name))
    zipFileName = file_name + "-" + time.strftime("%Y%m%d%H%M") + "-backup.zip"
    zipFileFullName = os.path.join(file_path, zipFileName)
    zf = zipfile.ZipFile(zipFileFullName, "w", zipfile.ZIP_DEFLATED)
    zf.write(file_full_path, file_name)
    zf.close()
    print("完成备份文件-{}".format(file_name))
    return zipFileName


def backupDir(dir_full_path):
    """
    文件夹备份,返回备份的文件名称。将文本在同目录备份成zip格式，命名为： 原文件全名-年月日时分-backup.zip。例如 abc.txt，备份为： abc.txt-202103141000-backup.zip
    :param dir_full_path: 目标文件夹的绝对路径
    :return: 返回备份的文件名称
    """
    assert os.path.isdir(dir_full_path), "zip备份的目标不是文件夹"
    dir_name = os.path.basename(dir_full_path)
    dir_path = os.path.dirname(dir_full_path)
    print("开始备份文件夹-{}".format(dir_name))
    zipDirName = dir_name + "-" + time.strftime("%Y%m%d%H%M") + "-backup.zip"
    zipDirFullName = os.path.join(dir_path, zipDirName)
    zf = zipfile.ZipFile(zipDirFullName, "w", zipfile.ZIP_DEFLATED)
    for tmp_dir_path, tmp_dir_names, tmp_file_names in os.walk(dir_full_path):
        f_path = tmp_dir_path.replace(dir_full_path, '')  # 这一句很重要，不replace的话，就从根目录开始复制
        f_path = f_path and f_path + os.sep or ''  # 实现当前文件夹以及包含的所有文件的压缩
        for filename in tmp_file_names:
            zf.write(os.path.join(tmp_dir_path, filename), os.path.join(dir_name, filename))
    zf.close()
    print("完成备份文件夹-{}".format(dir_name))
    return zipDirName


def copyFile(file1_path, file2_path, file2_name=None, show_info=False):
    """
    拷贝文件，从 file1 拷贝到 file2。具备对file1有效性的校验
    :param file1_path: 文件1全路径
    :param file2_path: 文件2路径
    :param file2_name: 文件2名称。如果该参数不为空，与 file2_path 组装成全路径
    :param show_info: 是否打印拷贝文件的路径信息
    :return:
    """
    assert os.path.isfile(file1_path), "拷贝文件的目标file1不是一个文件"
    if file2_name is None:
        shutil.copyfile(file1_path, file2_path)
        if show_info:
            print("拷贝文件，原路径:{}，目标路径:{}".format(file1_path, file2_path))
    else:
        target_full_path = os.path.join(file2_path, file2_name)
        shutil.copyfile(file1_path, target_full_path)
        if show_info:
            print("拷贝文件，原路径:{}，目标路径:{}".format(file1_path, target_full_path))



if __name__ == '__main__':
    test_dir = r'E:\aaa'
    import os

    dir1 = os.path.join(test_dir, 'dev')
    dir2 = os.path.join(test_dir, 'release')
    dir3 = os.path.join(test_dir, 'abc')

    dir4 = os.path.join(dir3, 'dev')
    dir5 = os.path.join(dir3, 'release')
    dir6 = os.path.join(dir3, 'def')

    dir_ls = [dir1, dir2, dir3, dir4, dir5, dir6]
    for this_dir in dir_ls:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)
        f1 = os.path.join(this_dir, 'test_1.py')
        f2 = os.path.join(this_dir, 'debug_1.py')
        f3 = os.path.join(this_dir, 'main.py')
        f4 = os.path.join(this_dir, 'main.pyc')
        f5 = os.path.join(this_dir, 'main.txt')
        f_ls = [f1, f2, f3, f4, f5]
        for file in f_ls:
            w = open(file, 'a', encoding='utf8')
            w.write(file)
            w.close()
    res_ls = []
    res_ls = search_files(test_dir, res_ls)
    print("不设条件的递归搜索结果：")
    for res in res_ls:
        print(' -- {}'.format(res))
    print()
    res_ls = []
    res_ls = search_files(
        test_dir,
        res_ls,
        corestr=['dev', 'release'],
        excludestr=['other', 'debug'],
        extname=['.pyc', '.py'],
        is_and=True)
    print("设置条件的递归搜索结果：")
    for res in res_ls:
        print(' -- {}'.format(res))
