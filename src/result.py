'''
    用于处理每次的schedule安排结果
'''

from matplotlib import pyplot as plt
from datetime import datetime

class Result:
    def __init__(self, cur_lst, machine_endtime, machine_owner, date):
        self.cur_lst = cur_lst
        self.machine_endtime = machine_endtime
        self.machine_owner = machine_owner
        self.date = date

    def present(self):
        for i in range(len(self.machine_owner)):
            for j in range(len(self.machine_owner[i])):
                time_slot = self.machine_endtime[i][j]
                plt.barh(i+1,time_slot[1]-time_slot[0],left=time_slot[0])
        plt.show()

    def save(self):
        f = open("../target/{0}".format(self.date.__format__('%Y.%m.%d %H-%M-%S')), "w")
        for i in range(len(self.machine_owner)):
            f.write("machine {0}\n".format(i+1))
            f.write("----------\n")
            for j in range(len(self.machine_owner[i])):
                time_slot = self.machine_endtime[i][j]
                cur_file = self.cur_lst[self.machine_owner[i][j][0]]
                start_state = self.cur_lst[self.machine_owner[i][j][0]].state[-1]
                cur_state_num = self.machine_owner[i][j][1] + int(start_state)
                cur_state = 'C' + str(cur_state_num)
                f.write('{0}s-{1}s with file {2} in state {3}\n'.format(time_slot[0],
                                                                        time_slot[1],
                                                                        cur_file.file_name, cur_state))
            f.write("---------------------------\n")
        f.close()

