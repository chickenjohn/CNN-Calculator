from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class ConvRadioModRecogNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(ConvRadioModRecogNetCalc, self).__init__(only_mac)

    '''
    Conv Net for Modulation Recog. 
    Source: http://arxiv.org/abs/1602.04105
    '''
    def ConvRadioModRecogNet(self, tensor):
        tensor = self.Conv2d(tensor, out_c=256, size=(3, 1), stride=(1, 1), padding=(2, 0))
        tensor = self.Conv2d(tensor, out_c=80, size=(3, 2), stride=(1, 0), padding=(2, 0))
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 256)
        tensor = self.Linear(tensor, 11)
        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.ConvRadioModRecogNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = ConvRadioModRecogNetCalc(only_mac=only_mac)
    calculator.calculate()
