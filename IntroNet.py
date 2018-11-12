from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class IntroNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(IntroNetCalc, self).__init__(only_mac)

    '''
    IntroNet Model.
    Source: http://arxiv.org/abs/1702.00832
    '''
    def IntroNet(self, tensor):
        tensor = self.Conv2d(tensor, out_c=128, size=(8, 2), stride=(1, 0), padding=(0, 0))
        tensor = self.MaxPool2d(tensor, size=(2, 1), stride=(2, 0))
        tensor = self.Conv2d(tensor, out_c=64, size=(16, 1), stride=(1, 0), padding=(0, 0))
        tensor = self.MaxPool2d(tensor, size=(2, 1), stride=(2, 0))
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 128)
        tensor = self.Linear(tensor, 64)
        tensor = self.Linear(tensor, 32)
        tensor = self.Linear(tensor, 10)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.IntroNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = IntroNetCalc(only_mac=only_mac)
    calculator.calculate()
