'''
    用于决策出每个时间点的最优策略
    输入是当前的采集文件队列，输出决策
'''

from random import randint
from random import sample
from src.gene import Gene

class Decider:
    def __init__(self, queue, population, pglobal, plocal, prandom, pcrossover, pmutation, iteration, tournamentk,
                 unsolved_queue,
                 threshold=None):
        self.machines_num = 3
        self.queue = queue
        # 分别是每个文件对应的状态和时间的矩阵、总共的状态数和每个状态在序列中的位置
        self.file_matrix, self.states_num, self.states_index = self.file_matrix()
        self.population = population
        self.pglobal = pglobal
        self.plocal = plocal
        self.prandom = prandom
        self.pcrossover = pcrossover
        self.pmutation = pmutation
        self.iteration = iteration
        self.threshold = threshold
        self.tournamentk = tournamentk
        self.unsolver_queue = unsolved_queue
        # 计算出states总数后根据它更新变异率
        self.pmutation = max(self.pmutation, 0.5/self.states_num)

    def global_search(self):
        def find_min_index(lst):
            min_value = min(lst)
            for i in range(len(lst)):
                if lst[i] == min_value:
                    return i
        chromosome = ['0' for _ in range(self.states_num)]
        random_list = self.get_random_file(self.file_matrix)
        # 初始化machine负载列表
        ms = [0 for _ in range(self.machines_num)]
        # 先处理可能的三个滞留文件的情况 先分配这些文件
        if self.unsolver_queue != [None, None, None]:
            for i in range(len(self.unsolver_queue)):
                if self.unsolver_queue[i] is not None:
                    file_index = self.unsolver_queue[i]
                    states_list = self.file_matrix[random_list[file_index]]
                    for s in range(len(states_list)):
                        if s == 0:
                            min_index = i
                        else:
                            min_index = find_min_index(ms)
                        ms[min_index] += states_list[s][1]
                        chromosome[self.states_index[states_list[s][0]]] = str(min_index + 1)
            # 这里不能单纯来一个pop一个 而需要集中pop 否则index将失效
            def filt(x):
                return x is not None
            sort_queue = list(self.unsolver_queue)
            sort_queue = list(filter(filt, sort_queue))
            sort_queue.sort(reverse=True)
            for i in sort_queue:
                random_list.pop(i)

        files_num = len(random_list)
        # 随机挑选工作（文件）且最后全部挑完
        for k in range(files_num):
            rand = randint(0,len(random_list)-1)
            states_list = self.file_matrix[random_list[rand]]
            for state in states_list:
                min_index = find_min_index(ms)
                ms[min_index] += state[1]
                chromosome[self.states_index[state[0]]] = str(min_index + 1)
            random_list.pop(rand)

        return ''.join(chromosome)

    def local_search(self):
        def find_min_index(lst):
            min_value = min(lst)
            for i in range(len(lst)):
                if lst[i] == min_value:
                    return i
        chromosome = ['0' for _ in range(self.states_num)]
        random_list = self.get_random_file(self.file_matrix)
        # 先处理可能的三个滞留文件的情况 先分配这些文件
        if self.unsolver_queue != [None, None, None]:
            for i in range(len(self.unsolver_queue)):
                if self.unsolver_queue[i] is not None:
                    ms = [0 for _ in range(self.machines_num)]
                    file_index = self.unsolver_queue[i]
                    states_list = self.file_matrix[random_list[file_index]]
                    for s in range(len(states_list)):
                        if s == 0:
                            min_index = i
                        else:
                            min_index = find_min_index(ms)
                        ms[min_index] += states_list[s][1]
                        chromosome[self.states_index[states_list[s][0]]] = str(min_index + 1)
            # 这里不能单纯来一个pop一个 而需要集中pop 否则index将失效
            def filt(x):
                return x is not None
            sort_queue = list(self.unsolver_queue)
            sort_queue = list(filter(filt, sort_queue))
            sort_queue.sort(reverse=True)
            for i in sort_queue:
                random_list.pop(i)

        files_num = len(random_list)
        # 随机挑选工作（文件）且最后全部挑完
        for k in range(files_num):
            ms = [0 for _ in range(self.machines_num)]
            rand = randint(0,len(random_list)-1)
            states_list = self.file_matrix[random_list[rand]]
            for state in states_list:
                min_index = find_min_index(ms)
                ms[min_index] += state[1]
                chromosome[self.states_index[state[0]]] = str(min_index + 1)
            random_list.pop(rand)

        return ''.join(chromosome)

    def random_search(self):
        chromosome = ""
        for i in range(self.states_num):
            rand = randint(1,3)
            chromosome = chromosome + str(rand)
        # 如果有未处理完的文件则在随机生成后需要更改指定机器
        if self.unsolver_queue != [None, None, None]:
            random_list = self.get_random_file(self.file_matrix)
            chromosome = list(chromosome)
            for i in range(len(self.unsolver_queue)):
                if self.unsolver_queue[i] is not None:
                    file_index = self.unsolver_queue[i]
                    states_list = self.file_matrix[random_list[file_index]]
                    chromosome[self.states_index[states_list[0][0]]] = str(i + 1)

        return ''.join(chromosome)

    def init_chromosome(self, pattern):
        def index_dict(lst):
            res = {}
            for i in range(len(lst)):
                res[lst[i]] = i
            return res
        if pattern == 'global':
            chromosome = self.global_search()
        elif pattern == 'local':
            chromosome = self.local_search()
        else:
            chromosome = self.random_search()
        # 下面开始随机生成染色体的后半部分
        random_list = self.get_random_file(self.file_matrix)
        index_dic = index_dict(random_list)
        tmp_hash = {}
        for key in self.file_matrix.keys():
            tmp_hash[key] = len(self.file_matrix[key])
        # 先处理未完成任务的情况
        unsolved_num = 0
        if self.unsolver_queue != [None, None, None]:
            for i in range(len(self.unsolver_queue)):
                if self.unsolver_queue[i] is not None:
                    unsolved_num += 1
                    index = self.unsolver_queue[i]
                    if index_dic[random_list[index]] + 1 < 10:
                        chromosome = chromosome + str(index_dic[random_list[index]] + 1)
                    else:
                        # 处理id大于9的情况，使用字母表示
                        chromosome = chromosome + chr(index_dic[random_list[index]] + 56)  # +1-10+65
                    # 弹出该文件的一个状态
                    tmp_hash[random_list[index]] -= 1
                    # 若没状态了，说明文件处理完了，直接把他从随机列表中移除
                    if tmp_hash[random_list[index]] == 0:
                        random_list.pop(index)
        # 正常流程
        for k in range(self.states_num - unsolved_num):
            rand = randint(0,len(random_list)-1)
            if index_dic[random_list[rand]]+1 < 10:
                chromosome = chromosome + str(index_dic[random_list[rand]]+1)
            else:
                # 处理id大于9的情况，使用字母表示
                chromosome = chromosome + chr(index_dic[random_list[rand]]+56)     # +1-10+65
            # 弹出该文件的一个状态
            tmp_hash[random_list[rand]] -= 1
            # 若没状态了，说明文件处理完了，直接把他从随机列表中移除
            if tmp_hash[random_list[rand]] == 0:
                random_list.pop(rand)
        # 所有的OS部分均为随机生成
        return chromosome

    def file_matrix(self):
        matrix = {}
        states_num = 0
        states_index = {}
        for job in self.queue:
            if job.records_remain is None:
                first_mat = job.records_num
                mat = job.records_num
            else:
                first_mat = job.records_remain
                mat = job.records_num
            if job.state == 'C1':
                matrix[job.file_name] = [(job.file_name + '_C1',first_mat/10**6), (job.file_name + '_C2',mat/20000),
                                         (job.file_name + '_C3',mat/200000)]
                states_index[job.file_name + '_C1'] = states_num
                states_index[job.file_name + '_C2'] = states_num + 1
                states_index[job.file_name + '_C3'] = states_num + 2
                states_num += 3
            elif job.state == 'C2':
                matrix[job.file_name] = [(job.file_name + '_C2', first_mat / 20000), (job.file_name + '_C3', mat / 200000)]
                states_index[job.file_name + '_C2'] = states_num
                states_index[job.file_name + '_C3'] = states_num + 1
                states_num += 2
            else:
                matrix[job.file_name] = [(job.file_name + '_C3', first_mat / 200000)]
                states_index[job.file_name + '_C3'] = states_num
                states_num += 1
        return matrix, states_num, states_index

    def get_random_file(self, matrix):
        lst = []
        for key in matrix.keys():
            lst.append(key)
        return lst

    def crossover(self, c1, c2):
        # 首先进行MS部分的交叉 使用UX均匀交叉
        # MS部分由于固定位置内容不会变所以不需要处理交叉后固定位置内容仍然不变
        rand_num = randint(1, self.states_num)
        rand_lst = sample(range(0,self.states_num), rand_num)
        child1 = ['0' for _ in range(self.states_num)]
        child2 = ['0' for _ in range(self.states_num)]
        # 先把不交叉的内容填上去 更有效率
        for i in rand_lst:
            child1[i] = c1[i]
            child2[i] = c2[i]
        # 补充填空
        for i in range(self.states_num):
            if child1[i] == '0':
                child1[i] = c2[i]
            if child2[i] == '0':
                child2[i] = c1[i]
        # 接下来对OS部分进行POX交叉 同理不需要特殊考虑未完成任务
        # 首先对工作进行随机分集
        set1 = set()
        set2 = set()
        for i in range(len(self.file_matrix)):
            rand = randint(1,2)
            if rand == 1:
                if i < 9:
                    set1.add(str(i+1))
                else:
                    set1.add(chr(i+56))
            else:
                if i < 9:
                    set2.add(str(i+1))
                else:
                    set2.add(chr(i+56))
        #先对child1进行POX
        child1.extend(['0' for _ in range(self.states_num)])
        for i in range(self.states_num,self.states_num*2):
            if c1[i] in set1:
                child1[i] = c1[i]
        i = self.states_num
        j = self.states_num
        # 使用c2对child1进行交叉生成
        while i < self.states_num*2 and j < self.states_num*2:
            while i < self.states_num*2 and c2[i] in set1:
                i += 1
            while j < self.states_num*2 and (child1[j] != '0'):
                j += 1
            # 全部校准到满足条件的位置 现在填空
            if i < self.states_num*2 and j < self.states_num*2:
                child1[j] = c2[i]
            i += 1
            j += 1
        #再对child2进行POX
        child2.extend(['0' for _ in range(self.states_num)])
        for i in range(self.states_num, self.states_num * 2):
            if c2[i] in set2:
                child2[i] = c2[i]
        i = self.states_num
        j = self.states_num
        # 使用c2对child1进行交叉生成
        while i < self.states_num * 2 and j < self.states_num * 2:
            while i < self.states_num * 2 and c1[i] in set2:
                i += 1
            while j < self.states_num * 2 and (child2[j] != '0'):
                j += 1
            # 全部校准到满足条件的位置 现在填空
            if i < self.states_num * 2 and j < self.states_num * 2:
                child2[j] = c1[i]
            i += 1
            j += 1
        # UX和POX部分全部完成，返回生成的两条子染色体
        return ''.join(child1), ''.join(child2)

    def mutation(self,chromosome):
        # 辅助邻域变异搜索算法
        def random_permutation(lst):
            my_lst = list(lst)
            res = []
            rand_index = randint(0,2)
            res.append(my_lst[rand_index])
            my_lst.pop(rand_index)
            rand_index = randint(0,1)
            res.append(my_lst[rand_index])
            my_lst.pop(rand_index)
            res.append(my_lst.pop())
            return res

        # 首先进行MS部分的变异
        rand_num = randint(1, self.states_num)
        rand_lst = sample(range(0, self.states_num), rand_num)
        # 将不能变的部分从rand_lst中剔除出去 同时统计数目供后面使用
        unsolved_num = 0
        if self.unsolver_queue != [None, None, None]:
            random_list = self.get_random_file(self.file_matrix)
            for i in range(len(self.unsolver_queue)):
                if self.unsolver_queue[i] is not None:
                    unsolved_num += 1
                    file_index = self.unsolver_queue[i]
                    states_list = self.file_matrix[random_list[file_index]]
                    if self.states_index[states_list[0][0]] in rand_lst:
                        rand_lst.pop(rand_lst.index(self.states_index[states_list[0][0]]))
        # 正式开始变异
        c_lst = list(chromosome)
        for i in rand_lst:
            rand = randint(1,self.machines_num)
            c_lst[i] = str(rand)
        # 然后进行OS部分的变异 使用邻域搜索变异（暂时使用随机选择）
        index_lst = sample(range(unsolved_num,self.states_num), 3)
        mat = [c_lst[self.states_num + index_lst[0]], c_lst[self.states_num + index_lst[1]],
               c_lst[self.states_num + index_lst[2]]]
        res = random_permutation(mat)
        for i in range(3):
            c_lst[self.states_num + index_lst[i]] = res[i]

        #两部分变异均完成
        return "".join(c_lst)

    def tournament(self, k, gene_lst):
        res = []
        for i in range(len(gene_lst)):
            # k表示每次随机取出多少个
            sample_lst = sample(gene_lst, k)
            max_fitness = 0
            index = -1
            for g in sample_lst:
                if max_fitness < g.fitness:
                    max_fitness = g.fitness
                    index = g
            # 此时index指向随机取出来的k个中的最优解
            res.append(Gene(g.chromosome,g.fitness))
        return res

    def decoder(self,chromosome):
        # 同时生成JM矩阵和T矩阵 顺便生成一个同维度的空矩阵
        j_matrix = []
        t_matrix = []
        process_matrix = []
        k = 0
        for job in self.file_matrix.keys():
            j_tmp = []
            t_tmp = []
            p_tmp = []
            for i in range(len(self.file_matrix[job])):
                p_tmp.append(0)
                process_lst = self.file_matrix[job]
                j_tmp.append(int(chromosome[k]))
                k += 1
                t_tmp.append(process_lst[i][1])
            j_matrix.append(list(j_tmp))
            t_matrix.append(list(t_tmp))
            process_matrix.append(list(p_tmp))
        #下面开始遍历染色体OS部分
        counter = [0 for _ in range(len(self.file_matrix))]
        # 用于记录每个机器被安排了工作的时间段
        machine_endtime = [[] for _ in range(self.machines_num)]
        # 对应于machine_endtime 用于记录每段时间分配给了谁
        machine_owner = [[] for _ in range(self.machines_num)]
        for i in range(self.states_num, self.states_num*2):
            # 统计出每个基因对应的工序（状态）的机器和时间
            if chromosome[i].isdigit():
                job = int(chromosome[i]) - 1
            else:
                job = ord(chromosome[i]) - 56
            jm = j_matrix[job][counter[job]]
            t = t_matrix[job][counter[job]]
            pre_process_finish_time = 0 if counter[job] == 0 else process_matrix[job][counter[job]-1]
            # 遍历当前工序选择的机器上的空闲时间段
            inserted = False
            for j in range(len(machine_endtime[jm-1])-1):
                if machine_endtime[jm-1][j][1] < machine_endtime[jm-1][j+1][0]:
                    ta = max(pre_process_finish_time, machine_endtime[jm-1][j][1])
                    if ta + t <= machine_endtime[jm-1][j+1][0]:
                        machine_endtime[jm-1].insert(j+1, (ta, ta+t))
                        machine_owner[jm-1].insert(j+1, (job, counter[job]))
                        process_matrix[job][counter[job]] = ta+t
                        inserted = True
                        break
            # 处理没有找到可插入空闲时间段的情况
            if not inserted:
                if len(machine_endtime[jm-1]) != 0:
                    ta = max(pre_process_finish_time, machine_endtime[jm-1][-1][1])
                    machine_endtime[jm-1].append((ta, ta+t))
                else:
                    ta = max(pre_process_finish_time, 0)
                    machine_endtime[jm - 1].append((ta, ta+t))
                process_matrix[job][counter[job]] = machine_endtime[jm - 1][-1][1]
                machine_owner[jm - 1].append((job, counter[job]))
            # 这里记得要加一个处理过的工序数量统计
            counter[job] += 1
        return machine_endtime, machine_owner

    def evaluate_chromosome(self, chromosome):
        machine_endtime, machine_owner = self.decoder(chromosome)
        # 已经完成了解码 接下来进行适应度评估
        # 这个for防止任务较少时有机器没有工作处理的情况
        for m in machine_endtime:
            if len(m) == 0:
                m.append((0,0))
        system_last_machine = max(machine_endtime,key=lambda x:x[-1][1])
        system_endtime = system_last_machine[-1][1]
        fitness = 100 / system_endtime
        return fitness

    def exec(self):
        gene_lst = []
        # 全局搜索随机生成
        for k in range(int(self.population*self.pglobal)):
            chromosome = self.init_chromosome('global')
            fitness = self.evaluate_chromosome(chromosome)
            g = Gene(chromosome, fitness)
            gene_lst.append(g)
        # 局域搜索随机生成
        for k in range(int(self.population*self.plocal)):
            chromosome = self.init_chromosome('local')
            fitness = self.evaluate_chromosome(chromosome)
            g = Gene(chromosome, fitness)
            gene_lst.append(g)
        # 随机搜索随机生成
        for k in range(int(self.population*self.prandom)):
            chromosome = self.init_chromosome('random')
            fitness = self.evaluate_chromosome(chromosome)
            g = Gene(chromosome, fitness)
            gene_lst.append(g)
        # 接下来就拥有了固定数量的染色体种群k
        t = 0
        best_fitness_gene = max(gene_lst,key=lambda x:x.fitness)
        best_fitness = best_fitness_gene.fitness
        while t < self.iteration and (self.threshold is None or
                                       (self.threshold is not None and best_fitness > self.threshold)):
            # 锦标赛法生成下一代种群
            gene_lst = self.tournament(self.tournamentk, gene_lst)
            # 每两个染色体进行配对 以一定概率交叉
            for i in range(self.population//2):
                p = randint(1,1000) / 1000
                if p <= self.pcrossover:
                    f1 = gene_lst[2*i]
                    f2 = gene_lst[2*i+1]
                    child1,child2 = self.crossover(f1.chromosome, f2.chromosome)
                    fitness1 = self.evaluate_chromosome(child1)
                    fitness2 = self.evaluate_chromosome(child2)
                    gene1 = Gene(child1, fitness1)
                    gene2 = Gene(child2, fitness2)
                    choosing_pool = [gene1, gene2, f1, f2]
                    # 将父子两代四个染色体按适应度从大到小排序，选出最优的两个
                    choosing_pool.sort(key=lambda x:x.fitness,reverse=True)
                    gene_lst[2*i] = choosing_pool[0]
                    gene_lst[2*i+1] = choosing_pool[1]
            # 每个染色体以一定的概率进行变异
            for i in range(self.population):
                p = randint(1,1000) / 1000
                if p <= self.pmutation:
                    child = self.mutation(gene_lst[i].chromosome)
                    fitness = self.evaluate_chromosome(child)
                    gene_lst[i] = Gene(child, fitness)

            best_fitness_gene = max(gene_lst, key=lambda x: x.fitness)
            best_fitness = best_fitness_gene.fitness
            print("Iteration No.{0}  best_fitness: {1}".format(t+1, best_fitness))
            t += 1

        # 此时得到的种群即最终结果 选出最优适应度个体
        max_f = 0
        index = -1
        for i in range(self.population):
            if max_f < gene_lst[i].fitness:
                max_f = gene_lst[i].fitness
                index = i
        best_individual = gene_lst[index]
        machine_endtime, machine_owner = self.decoder(best_individual.chromosome)
        return machine_endtime, machine_owner


# if __name__ == '__main__':
#     from src.collection_file import CollectionFile
#     from datetime import datetime
#     f1 = CollectionFile(1, 'A1', 'A1_F1', 105747, datetime(2015,6,18,10,19,22))
#     f2 = CollectionFile(2, 'A1', 'A1_F2', 606909, datetime(2015, 6, 18, 10, 19, 22))
#     f3 = CollectionFile(3, 'A2', 'A2_F1', 1177054, datetime(2015, 6, 18, 10, 19, 22))
#     f4 = CollectionFile(4, 'A2', 'A2_F2', 510757, datetime(2015, 6, 18, 10, 19, 34))
#     f1.records_remain = 5747
#     f2.records_remain = 600000
#     f3.records_remain = 1177050
#     queue = [f1,f2,f3,f4]
#     decider = Decider(queue,4,0.5,0.25,0.25,1.0,0.25,100,2,[0,1,2])
#     print(decider.crossover('121212321211123343411422','131232323111123343421421'))