from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt, pow
import numpy as np

# train = 'WE_data/data_train.csv'
# test = 'WE_data/data_test.csv'
train = 'WE_data/train_local.csv'
test = 'WE_data/test_local.csv'


submission = 'submission_{0}.csv'.format(datetime.now().strftime('%B_%d_%H:%M:%S'))  # path of to be outputted submission file


alpha = .17
beta = 1
L1 = 1
L2 = 1

D = 2 ** 22
epoch = 5
holdafter = None
holdout = 100
loss=0.



class ftrl_proximal(object):

    def __init__(self, alpha, beta, L1, L2, D):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2


        self.D = D

        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):

        yield 0

        for index in x:
            yield index


    def predict(self, x):

        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}


        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.

            if sign * z[i] <= L1:

                w[i] = 0.
            else:

                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]


        self.w = w

        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):

        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w


        g = p - y

        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def seloss(p, y):

    return pow(p-y,2)


def data(path, D):

    n = 0
    for row in DictReader(open(path)):

        # process clicks
        y = 0.
        if 'Click' in row:
            if row['Click'] == '1':
                y = 1.
            del row['Click']

        x = []
        for key in row:
            value = row[key]
            if key == 'UserTags':
                vs = value.split('_')
                vs.sort()
                for v in vs:
                    index = abs(hash(key + '_' + v)) % D
                    x.append(index)
            else:
                index = abs(hash(key + '_' + value)) % D
                x.append(index)

        n = n + 1
        yield n,x,y


start = datetime.now()

learner = ftrl_proximal(alpha, beta, L1, L2, D)


for e in xrange(epoch):
    se = 0.
    count = 0
    for n, x, y in data(train, D):  # data is a generator

        p = learner.predict(x)

        # se += seloss(p, y)
        # count += 1
        # learner.update(x, p, y)

        if (holdout and n % holdout == 0):
            se += seloss(p, y)
            count += 1
        else:
            learner.update(x, p, y)

        # if n % 100000 == 0:
        #     print('Dealing with sample {0}'.format(n))

    print('Epoch '+str(e)+'  finished, validation RMSE: '+ str(sqrt(se/count))+', elapsed time: '  +str(( datetime.now() - start)))


with open(submission, 'w') as outfile:
    outfile.write('Id,Prediction\n')
    t = 1
    for t, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (t, str(p)))
        t = t + 1
