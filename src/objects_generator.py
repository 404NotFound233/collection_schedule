'''
    用于从xlsx文件中生成信息类文件对象序列
'''

from src.collection_file import CollectionFile
import openpyxl

class Generator:
    # 读取xlsx文件
    def __init__(self, path):
        self.path = path
    def get_list_from_xlsx(self):
        lst = []
        xlsx = openpyxl.load_workbook(self.path)
        sheet = xlsx['Sheet1']
        for row in sheet.rows:
            record = [col.value for col in row]
            lst.append(record)
        lst = lst[2:]
        return lst

    # 从xlsx文件中读取的内容中生成对象序列
    def generate_queue(self):
        xlsx_lst = self.get_list_from_xlsx()
        ob_lst = []
        come_in_set = set()
        for file in xlsx_lst:
            id = int(file[0])
            info_class = file[1]
            file_name = file[2]
            records_num = int(file[3])
            come_in_date = file[4]
            come_in_set.add(come_in_date)
            cf = CollectionFile(id, info_class, file_name, records_num, come_in_date)
            ob_lst.append(cf)
        ob_lst.sort(key=lambda x:x.come_in_date)
        return ob_lst