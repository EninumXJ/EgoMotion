import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from model.twin_network import TwinNetwork, R3D
from model.loss import *
from model.proj import Projector
from dataset.ego_dataset import EgoMotionDataset
from MotionBERT.lib.model.DSTformer import DSTformer
from MotionBERT.lib.utils.tools import *
from functools import partial

if __name__=='__main__':
    config_path = 'config/MB_twin_network_finetune.yaml'
    args = get_config(config_path)
    checkpoints_path = '/home/litianyi/workspace/EgoMotion/checkpoints/exp01/best_epoch.pth'
    twin_net = R3D()
    proj = Projector(256, 51*10)
    model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        twin_net = nn.DataParallel(twin_net)
        proj = nn.DataParallel(proj)
        model_backbone = model_backbone.cuda()
        twin_net = twin_net.cuda()
        proj = proj.cuda()
    checkpoint = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['backbone'], strict=True)
    twin_net.load_state_dict(checkpoint['twin_net'], strict=True)
    proj.load_state_dict(checkpoint['proj'], strict=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    normalize])

    egodata = EgoMotionDataset(dataset_path=args.data_root,
                         config_path=args.config,
                         image_tmpl=args.image_tmpl,
                         transform=img_transforms,
                         clip_length=args.clip_len)
    # test_loader = DataLoader(egodata,
    #                           batch_size=1, shuffle=True,
    #                           num_workers=4, pin_memory=False)
    input, gt = egodata[20]
    input = torch.tensor(input).cuda().unsqueeze(0).permute(0,2,1,3,4)
    gt = torch.tensor(gt).cuda().unsqueeze(0)
    print("input: ", input)
    print("ground truth: ", gt)
    representation = twin_net(input)
    print("represent: ", representation)
    embeddings = proj(representation).reshape(1, -1, 17, 3)
    print("embedding: ", embeddings)
    output = model_backbone(embeddings)
    print("output: ", output)
    pred = output.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    print("err1: ", mpjpe(pred, gt)*100)
