'''
    中心指挥类
    用于根据xlsx信息模拟整个流程
    调用其他所有工具
'''

from src.objects_generator import Generator
from src.decider import Decider
from src.result import Result
from datetime import datetime

class Commander:
    def __init__(self, xlsx_path):
        self.xlsx_path = xlsx_path
    def exec(self):
        print('Loading schedule file ……')
        generator = Generator(self.xlsx_path)
        # schedule是按时间顺序排序过后的采集文件集
        schedule = generator.generate_queue()
        print('Successfully loaded schedule!')
        k = 0
        print('Starting experiment ……')
        cur_lst = []
        while k < len(schedule):
            # 删除已经完成的任务 添加新的任务
            if k != 0:
                self.refresh_queue(machine_endtime, machine_owner, schedule[k].come_in_date, cur_lst)
            cur_lst.append(schedule[k])
            k += 1
            while k < len(schedule) and schedule[k].come_in_date == schedule[k-1].come_in_date:
                cur_lst.append(schedule[k])
                k += 1
            # 此时cur_lst包含了某个时段进入的所有采集文件
            print('{0} new files come in ……'.format(cur_lst[-1].come_in_date))
            decider = Decider(queue=cur_lst,
                              population=200, pglobal=0.6,plocal=0.05,prandom=0.35,
                              pcrossover=0.8,pmutation=0.01,iteration=100,tournamentk=100)
            machine_endtime, machine_owner = decider.exec()
            result = Result(cur_lst, machine_endtime, machine_owner,cur_lst[-1].come_in_date)
            result.save()
            result.present()
            # 上方针对一次新的文件输入完成了调度安排展示和持久化


    def refresh_queue(self, machine_endtime, machine_owner, cur_time, cur_lst):
        start_time = cur_lst[0].come_in_date
        relative_time = cur_time - start_time
        relative_second = relative_time.seconds
        completed_index = []
        for i in range(len(machine_endtime)):
            for j in range(len(machine_endtime[i])):
                time_slot = machine_endtime[i][j]
                # 如果该安排结束时间早于现在的时间 也就是在新内容出现时已经完成
                if time_slot[1] < relative_second:
                    cur_file = cur_lst[machine_owner[i][j][0]]
                    cur_state = cur_lst[machine_owner[i][j][0]].state
                    if cur_state == 'C1':
                        cur_file.state = 'C2'
                    elif cur_state == 'C2':
                        cur_file.state = 'C3'
                    elif cur_state == 'C3':
                        cur_file.state = 'Completed'
                        completed_index.append(machine_owner[i][j][0])
                else:
                    break
        # 所有任务执行状态更新完毕 下面从cur_lst中去除完成了的任务
        completed_index.sort(reverse=True)
        for ci in completed_index:
            cur_lst.pop(ci)

if __name__ == '__main__':
    commander = Commander('../data/schedule.xlsx')
    commander.exec()