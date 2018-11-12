from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class WiiNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(WiiNetCalc, self).__init__(only_mac)

    '''
    Wireless Interference Identification Net Model.
    Source: http://arxiv.org/abs/1703.00737
    '''
    def WiiNet(self, tensor):
        tensor = self.Conv2d(tensor, out_c=8, size=(3, 1), stride=(1, 1), padding=(0, 0))
        tensor = self.Conv2d(tensor, out_c=16, size=(3, 2), stride=(1, 0), padding=(0, 0))
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 64)
        tensor = self.Linear(tensor, 15)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.WiiNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = WiiNetCalc(only_mac=only_mac)
    calculator.calculate()
