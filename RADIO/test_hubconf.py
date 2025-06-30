import sys
import torch
from hubconf import radio_model
from thop import profile
from thop import clever_format
from timm.models import registry


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total {total_num}, Trainable {trainable_num}')
    # return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    # model = 'vit_huge_patch16_224'
    model = radio_model(version='radio_v2.5-h').cuda()
    get_parameter_number(model)
    # state_dict = model.state_dict()
    # torch.save(state_dict, './weights/pretrained/weights.pth')

    # flops, params = profile(model, inputs=(x,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs, params)




    # with torch.no_grad():
    #     y = model(x)

        # op, (int0,) = model.forward_intermediates(x, indices=[-1], output_fmt='NLC', aggregation='sparse')
        #
        # diff = (op.features - int0).norm()
        # print(f'Output diff: {diff.item():.8f}')
        #
        # y_int1 = model.forward_intermediates(x, indices=[1, 5, 7], output_fmt='NCHW')
        # y_int2 = model.forward_intermediates(x, indices=[2, 4, 6], output_fmt='NLC')
        # y_int3 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True)
        # y_int4 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True, norm_alpha_scheme='pre-alpha')
        # y_int5 = model.forward_intermediates(x, indices=[3, 5, 7], return_prefix_tokens=True, output_fmt='NCHW', aggregation='dense', intermediates_only=True, norm_alpha_scheme='none')
        # pass
