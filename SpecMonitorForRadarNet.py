from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator

class SpecMonitorForRadarNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(SpecMonitorForRadarNetCalc, self).__init__(only_mac)

    '''
    Spec. Monitor for Radar Net Model.
    Source: http://arxiv.org/abs/1705.00462
    '''
    def SpecMonitorForRadarNet(self, tensor):
        tensor = self.Conv2d(tensor, out_c=48, size=(11, 11), stride=(1, 1), padding=(0, 0))
        tensor = self.MaxPool2d(tensor, size=(11, 11), stride=(1, 1), padding=(0, 0))
        tensor = self.Conv2d(tensor, out_c=128, size=(5, 5), stride=(1, 1), padding=(0, 0))
        tensor = self.MaxPool2d(tensor, size=(5, 5), stride=(1, 1), padding=(0, 0))
        tensor = self.Conv2d(tensor, out_c=192, size=(3, 3), stride=(1, 1), padding=(0, 0))
        tensor = self.Conv2d(tensor, out_c=192, size=(3, 3), stride=(1, 1), padding=(0, 0))
        tensor = self.Conv2d(tensor, out_c=128, size=(3, 3), stride=(1, 1), padding=(0, 0))
        tensor = self.Flatten(tensor)
        tensor = self.Linear(tensor, 1024)
        tensor = self.Linear(tensor, 1024)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 64, 64)
        tensor = self.SpecMonitorForRadarNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

if __name__ == '__main__':
    only_mac = False

    calculator = SpecMonitorForRadarNetCalc(only_mac=only_mac)
    calculator.calculate()
