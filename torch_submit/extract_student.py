import torch

ckpt = torch.load('./data/out/latest.pth', 'cpu')
new_ckpt = {}
for k,v in ckpt['state_dict'].items():
    if 'student_net' in k:
        new_ckpt[k.replace('student_net.','')] = v

torch.save(new_ckpt, './data/out/latest-converted.pth')
