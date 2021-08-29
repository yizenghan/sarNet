import numpy as np

TH=0.99
class CalculationCalculator(object):
    def __init__(self, model='res101', sparsity_dict=None):
        self.resolution = 224
        self.h = self.resolution
        self.w = self.resolution
        self.model = model
        self.DCN = False

        assert sparsity_dict is not None
        self.sparsity_dict = sparsity_dict
        self.lg_kernel = 5
        self.mask_kernel = 3
        self.bind23 = True

        assert self.model in ['res34', 'res50', 'res101']
        if self.model == 'res34':
            self.block = 'Basic'
            self.blocks = [3, 4, 6, 3]
        if self.model == 'res50':
            self.block = 'Bottleneck'
            self.blocks = [3, 4, 6, 3]
        elif self.model == 'res101':
            self.block = 'Bottleneck'
            self.blocks = [3, 4, 23, 3]
        self.planes = [64, 128, 256, 512]
        self.strides = [4, 8, 16, 32]

        self.reset()

    def reset(self):
        self.calculation = dict()
        self.whole_cal = 0

    def conv_cal(self, kernel, inplane, outplane, h, w):
        return kernel * kernel * inplane * outplane * h * w

    def bottleneck_block_cal(self, inplane, plane, stride, sparsity):
        st1, st2, st3 = stride
        sp1, sp2, sp3 = sparsity
        c1 = self.conv_cal(1, inplane, plane, self.h // st1, self.w // st1) * sp1
        c2 = self.conv_cal(3, plane, plane, self.h // st2, self.w // st2) * sp2
        c3 = self.conv_cal(1, plane, plane * 4, self.h // st3, self.w // st3) * sp3
        lt = self.conv_cal(1, inplane, plane * 4, self.h // st3, self.w // st3) if inplane != plane * 4 else 0
        m1 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st1, self.w // st1)
        m2 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st2, self.w // st2)
        m3 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st3, self.w // st3) if not self.bind23 else 0
        i1 = self.conv_cal(self.lg_kernel, plane, 1, self.h // st1, self.w // st1) * sp1 * (1 - sp1)
        i2 = self.conv_cal(self.lg_kernel, plane, 1, self.h // st2, self.w // st2) * sp2 * (1 - sp2) if not self.bind23 else 0
        i3 = self.conv_cal(self.lg_kernel, plane * 4, 1, self.h // st3, self.w // st3) * sp3 * (1 - sp3)
        return c1 + c2 + c3 + m1 + m2 + m3 + i1 + i2 + i3 + lt

    def basic_block_cal(self, **kwargs):
        # method = 'LCCL'
        method = 'Full'
        if method == 'LCCL':
            return self.basic_block_cal_LCCL(**kwargs)

        if method == 'Full':
            return self.basic_block_cal_Full(**kwargs)

    def basic_block_cal_LCCL(self, inplane, plane, stride, sparsity):
        st1, st2 = stride
        sp1, sp2 = sparsity
        c1 = self.conv_cal(3, inplane, plane, self.h // st1, self.w // st1)
        c2 = self.conv_cal(3, plane, plane, self.h // st2, self.w // st2)
        lt = self.conv_cal(1, inplane, plane, self.h // st2, self.w // st2) if inplane != plane else 0
        m1 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st1, self.w // st1)
        m2 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st2, self.w // st2)

        ret = lt
        if sp1 < TH:
            ret += c1 * sp1
            ret += m1
        else:
            ret += c1

        if sp2 < TH:
            ret += c2 * sp2
            ret += m2
        else:
            ret += c2

        return ret

    def basic_block_cal_Full(self, inplane, plane, stride, sparsity):
        st1, st2 = stride
        sp1, sp2 = sparsity
        c1 = self.conv_cal(3, inplane, plane, self.h // st1, self.w // st1)
        c2 = self.conv_cal(3, plane, plane, self.h // st2, self.w // st2)
        lt = self.conv_cal(1, inplane, plane, self.h // st2, self.w // st2) if inplane != plane else 0
        m1 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st1, self.w // st1)
        m2 = self.conv_cal(self.mask_kernel, inplane, 1, self.h // st2, self.w // st2)
        i1 = self.conv_cal(self.lg_kernel, plane, 1, self.h // st1, self.w // st1) * sp1 * (1 - sp1)
        i2 = self.conv_cal(self.lg_kernel, plane, 1, self.h // st2, self.w // st2) * sp2 * (1 - sp2)

        ret = lt #c1 * sp1 + c2 * sp2 + lt
        if sp1 < TH:
            ret += c1 * sp1
            ret += m1 + i1
        else:
            ret += c1

        if sp2 < TH:
            ret += c2 * sp2
            ret += m2 + i2
        else:
            ret += c2

        return ret

        # if sparsity < 0.999:
        #     return c1 + c2 + m1 + m2 + i1 + i2 + lt
        # else:
        #     return c1 + c2 + lt

    def get_cal(self):
        print('TH: {0}'.format(TH))
        if self.block == 'Basic':
            _cal_func = self.basic_block_cal
        else:
            _cal_func = self.bottleneck_block_cal

        self.calculation['conv1'] = self.conv_cal(7, 3, 64, self.h // 2, self.w // 2)
        print(self.calculation['conv1'])
        self.whole_cal += self.calculation['conv1']

        for idx_stage, blocks in enumerate(self.blocks):
            for idx_block in range(blocks):
                key = f'Res{idx_stage + 1}_Block{idx_block + 1}'
                if self.block == 'Basic':
                    stride = (self.strides[idx_stage], self.strides[idx_stage])
                    sparsity = (self.sparsity_dict[key]['Conv1'], self.sparsity_dict[key]['Conv2'])
                else:
                    stride = (
                        self.strides[idx_stage] // 2 if (idx_stage != 0 and idx_block == 0) else self.strides[idx_stage],
                        self.strides[idx_stage],
                        self.strides[idx_stage])
                    sparsity = (
                        self.sparsity_dict[key]['Conv1'],
                        self.sparsity_dict[key]['Conv2'],
                        self.sparsity_dict[key]['Conv2'] if not self.sparsity_dict[key].get('Conv3') else self.sparsity_dict[key]['Conv3'])

                if idx_block == 0:
                    inplane = self.planes[0] if idx_stage == 0 else self.planes[idx_stage - 1]
                else:
                    inplane = self.planes[idx_stage]
                self.calculation[key] = _cal_func(
                    inplane=inplane,
                    plane=self.planes[idx_stage],
                    stride=stride,
                    sparsity=sparsity,
                )
                print(f'{idx_block} - Stride {stride} - Sparsity {sparsity} - Calculation {self.calculation[key]:,}')
                self.whole_cal += self.calculation[key]

        print(f'Total Computation - {self.whole_cal:,}')


# if __name__ == '__main__':
#     CC = CalculationCalculator()