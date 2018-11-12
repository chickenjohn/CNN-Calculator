from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class EndToEndNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(EndToEndNetCalc, self).__init__(only_mac)

    '''
    End-to-end learning for Spectrum Sensing model.
    Source: http://arxiv.org/abs/1712.03987
    '''
    def EndToEndNet(self, tensor):
        tensor = self.Conv2d(tensor, out_c=256, size=(3, 1), stride=(1, 1), padding=(1, 0))
        tensor = self.Conv2d(tensor, out_c=80, size=(3, 2), stride=(1, 0), padding=(1, 0))
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 256)
        tensor = self.Linear(tensor, 11)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.EndToEndNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = EndToEndNetCalc(only_mac=only_mac)
    calculator.calculate()
