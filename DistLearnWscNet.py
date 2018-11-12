from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class DistLearnWscNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(DistLearnWscNetCalc, self).__init__(only_mac)

    '''
    Dist. Learning for Wireless Signal Classif. Net Model.
    Source: http://arxiv.org/abs/1707.08908
    '''
    def DistLearnWscNet(self, tensor):
        tensor = self.LSTM(tensor, 128, 1)
        tensor = self.LSTM(tensor, 128, 0)
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 11)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.DistLearnWscNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = DistLearnWscNetCalc(only_mac=only_mac)
    calculator.calculate()
