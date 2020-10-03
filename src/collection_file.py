'''
    信息类的信息文件类
    是执行器处理的对象的类
'''

class CollectionFile:
    id = 0
    info_class = ""
    file_name = ""
    records_num = 0
    state = ""
    come_in_date = None
    refresh_date = None

    def __init__(self, id, info_class, file_name, records_num, come_in_date):
        self.id = id
        self.info_class = info_class
        self.file_name = file_name
        self.records_num = records_num
        self.state = 'C1'
        self.come_in_date = come_in_date
        self.refresh_date = self.come_in_date

    #获取当前采集文件当前状态下处理要多久
    def get_time_cost(self):
        efficiency = [1000, 20, 200]
        choice_factor = int(self.state[1]) - 1
        return self.records_num / efficiency[choice_factor]
