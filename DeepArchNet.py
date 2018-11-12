from __future__ import print_function
from DNNCalculator import Tensor, DNNCalculator


class DeepArchNetCalc(DNNCalculator):
    def __init__(self, only_mac=True):
        super(DeepArchNetCalc, self).__init__(only_mac)

    '''
    Deep Arch for Modulation Recognition Net Model.
    Source: https://ieeexplore.ieee.org/abstract/document/7920754
    '''

    def DeepArchNet(self, tensor):
        tensor1 = self.Conv2d(tensor, out_c=50, size=(
            8, 1), stride=(1, 1), padding=0)
        tensor2 = self.Conv2d(tensor1, out_c=50, size=(
            8, 1), stride=(1, 1), padding=0)
        tensor3 = self.Conv2d(tensor2, out_c=50, size=(
            8, 1), stride=(1, 1), padding=0)
        tensor1 = Tensor(tensor1.h, tensor1.c, tensor1.w)
        tensor3 = Tensor(tensor3.h, tensor3.c, tensor3.w)
        tensor = self.Concat([tensor1, tensor3])
        tensor_after_concat = Tensor(1, tensor.h*tensor.w, tensor.c)
        tensor = self.LSTM(tensor_after_concat, 11, 0)

        return tensor

    def calculate(self):
        tensor = Tensor(1, 128, 2)
        tensor = self.DeepArchNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))


if __name__ == '__main__':
    only_mac = False

    calculator = DeepArchNetCalc(only_mac=only_mac)
    calculator.calculate()
