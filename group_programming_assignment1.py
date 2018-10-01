import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import pickle as pkl

class group_programming_assignment:
    def __init__(self):
        self.ns = [4, 8, 16, 32, 64, 128, 256, 512]
        self.cals = [self.add, self.subtract, self.multiply, self.divide]

    def get_ndigit(self, n, len=1000):
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        return np.array([random.randint(range_start, range_end) for i in range(len)])

    def add(self, n):
        input1 = self.get_ndigit(n)
        input2 = self.get_ndigit(n)
        return input1 + input2

    def subtract(self, n):
        input1 = self.get_ndigit(n)
        input2 = self.get_ndigit(n)
        return input1 - input2

    def multiply(self, n):
        input1 = self.get_ndigit(n)
        input2 = self.get_ndigit(n)
        return input1 * input2

    def divide(self, n):
        input1 = self.get_ndigit(n)
        input2 = self.get_ndigit(n)
        return input1 / input2

    def performance_test(self):
        len_ns = len(self.ns)
        len_cals = len(self.cals)

        performance = np.zeros((len_ns, len_cals))
        index = []
        for i in range(len_ns):
            idx = self.ns[i]
            index.append(idx)
            columns = []
            for j in range(len_cals):
                start_time = time.time()
                self.cals[j](idx)
                performance[i, j] = (time.time() - start_time) * 1e3
                columns.append(self.cals[j].__name__)

        return pd.DataFrame(data=performance, columns=columns, index=index)

    def plot(self, bin, hist, label):
        binwidth = bin[1] - bin[0]
        plt.bar(bin[:-1], hist, label=label, alpha=.5, width=binwidth)

    def statistics(self, df_res):
        stats = []
        index = []
        for n in self.ns:
            plt.figure(figsize=(12, 8))
            for cal in self.cals:
                values = df_res.loc[n, cal.__name__].values
                hist, bin = np.histogram(values, bins=100)
                stats.append([values.mean(), values.std()])
                index.append('{},{}'.format(n, cal.__name__))

                self.plot(bin, hist, 'func: {}'.format(cal.__name__))

            plt.grid()
            plt.legend()
            plt.title('n={}'.format(n))
            plt.show()
        df_stat = pd.DataFrame(data=stats, columns=['mean', 'std'], index=index)
        print(df_stat)

if __name__ == '__main__':
    g = group_programming_assignment()
    N = 100
    df = g.performance_test()
    df.to_excel('performace.xlsx')
    # cost_time_mean = df.mean(axis=1)

    for i, col in enumerate(df.columns):
        plt.subplot(2, 2, i + 1)
        plt.plot(df.index, df[col])
        plt.grid()
        plt.xlabel('n-digits')
        plt.ylabel('Average Cost Time (ms)')
        plt.title(col)
    plt.show()

    # dfs = []
    # for i in range(100):
    #     df = g.performance_test()
    #     df['n'] = i
    #     dfs.append(df)
    # df_res = pd.concat(dfs)

    # with open('temp.cha', 'wb') as f:
    #     pkl.dump(df_res, f)
    #
    # with open('temp.cha', 'rb') as f:
    #     df_res = pkl.load(f)
    # g.statistics(df_res)

