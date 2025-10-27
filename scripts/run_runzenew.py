

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
from tqdm import tqdm

def vis_depth_gray(depth, min_val=None, max_val=None, invalid_thres=np.inf, other_output={}):
    """
    生成灰度图形式的深度图可视化

    @depth: np.array(H, W), float32，单位：米
    @min_val: 可选，最小深度值
    @max_val: 可选，最大深度值
    @invalid_thres: 超过该阈值或 <= 0 的值视为无效
    @other_output: 可选字典，用于输出 min/max
    """
    depth = depth.copy()
    H, W = depth.shape[:2]
    invalid_mask = (depth >= invalid_thres) | (depth <= 0) | np.isnan(depth)

    if (invalid_mask == 0).sum() == 0:
        other_output['min_val'] = None
        other_output['max_val'] = None
        return np.zeros((H, W), dtype=np.uint8)

    if min_val is None:
        min_val = depth[~invalid_mask].min()
    if max_val is None:
        max_val = depth[~invalid_mask].max()

    other_output['min_val'] = min_val
    other_output['max_val'] = max_val

    norm_depth = ((depth - min_val) / (max_val - min_val)).clip(0, 1) * 255
    gray = norm_depth.astype(np.uint8)
    gray[invalid_mask] = 0

    return gray  # 返回灰度图 (H, W)，uint8
def disparity_to_depth(disparity, focal_length, baseline, scale,invalid_disparity_thres=1e-6):
    """
    从视差图计算深度图
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_mask = disparity > invalid_disparity_thres
    depth[valid_mask] = (focal_length * baseline * scale) / disparity[valid_mask]
    return depth

def vis_depth(depth, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
    depth = depth.copy()
    H, W = depth.shape[:2]
    invalid_mask = (depth >= invalid_thres) | (depth <= 0) | np.isnan(depth)

    if (invalid_mask == 0).sum() == 0:
        other_output['min_val'] = None
        other_output['max_val'] = None
        return np.zeros((H, W, 3), dtype=np.uint8)

    if min_val is None:
        min_val = depth[~invalid_mask].min()
    if max_val is None:
        max_val = depth[~invalid_mask].max()

    other_output['min_val'] = min_val
    other_output['max_val'] = max_val

    norm_depth = ((depth - min_val) / (max_val - min_val)).clip(0, 1) * 255

    if cmap is None:
        vis = cv2.applyColorMap(norm_depth.astype(np.uint8), color_map)[..., ::-1]  # BGR to RGB
    else:
        vis = cmap(norm_depth.astype(np.uint8))[..., :3] * 255

    vis[invalid_mask] = 0
    return vis.astype(np.uint8)

root1 = "/share/xurunze-local/FoundationPose/1014"
bags = sorted(os.listdir(root1))
root = root1 + bags[0]

if __name__=="__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir', default=f'{root}/left_rectified', type=str)
    parser.add_argument('--right_dir', default=f'{root}/right_rectified', type=str)
    parser.add_argument('--intrinsic_file', default=f'/share/xurunze-local/code_x2robot/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'/share/xurunze-local/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{root}/depth/', type=str, help='the directory to save results')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale', default=1, type=float, help='scale factor for input images')

    args = parser.parse_args()
    scale = args.scale

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    with open(args.intrinsic_file, 'r') as f:
        data = f.readlines()
        fx = float(data[0].split()[0])  # 假设第一行是fx
        baseline = float(data[1].split()[0])  # 假设第二行是baseline

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    print(torch.cuda.device_count())
    model = nn.DataParallel(model).cuda() 
    #model.cuda()
    model.eval()

    for bag in bags:
        if bag[0] != '@' or bag[-4:] == '.zip':
            continue
        else:
            bag_path = os.path.join(root1, bag)
            args.left_dir = f'{bag_path}/left_rectified'
            args.right_dir = f'{bag_path}/right_rectified'
            args.out_dir = f'{bag_path}/depth'
            os.makedirs(args.out_dir, exist_ok=True)
            left_files = sorted([f for f in os.listdir(args.left_dir) if f.endswith("png")])
            right_files = sorted([f for f in os.listdir(args.right_dir) if f.endswith("png")])

            if not left_files or not right_files:
                print("No images found in input directories!")
                exit(1)

            if left_files != right_files:
                print("Left and right image lists don't match exactly. Attempting pair by common names.")

            
            
            B = 8
            num_batches = (len(left_files) + B - 1) // B

            for idx in tqdm(range(0, len(left_files), B), desc=f"Processing bag {bag}"):
                print(f"Processing batch {idx} ...")
                batch_left_files = left_files[idx:idx+B]
                batch_right_files = right_files[idx:idx+B]
                imgs0, imgs1, sizes = [], [], []
                t0 = time.time()

                for left_file, right_file in zip(batch_left_files, batch_right_files):
                    left_path = os.path.join(args.left_dir, left_file)
                    right_path = os.path.join(args.right_dir, right_file)

                    img0 = imageio.imread(left_path)
                    img1 = imageio.imread(right_path)
                    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
                    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
                    H, W = img0.shape[:2]
                    sizes.append((H, W))

                    imgs0.append(torch.as_tensor(img0).float().permute(2, 0, 1))
                    imgs1.append(torch.as_tensor(img1).float().permute(2, 0, 1))

                imgs0 = torch.stack(imgs0)
                imgs1 = torch.stack(imgs1)

                # 判断 batch 是否小于 GPU 数量，或者 batch size == 1
                if imgs0.shape[0] < torch.cuda.device_count() or imgs0.shape[0] == 1:
                    disp_list = []
                    
                    for i in range(imgs0.shape[0]):
                        img0_i, img1_i = imgs0[i:i+1].cuda(), imgs1[i:i+1].cuda()
                        padder = InputPadder(img0_i.shape, divis_by=32, force_square=False)
                        img0_i, img1_i = padder.pad(img0_i, img1_i)
                        with torch.cuda.amp.autocast(True):
                            disp_i = model.module(img0_i, img1_i, iters=args.valid_iters, test_mode=True)
                        disp_i = padder.unpad(disp_i.float())
                        disp_list.append(disp_i.cpu())
                    disp = torch.cat(disp_list, dim=0).numpy()
                else:
                    # 多卡 batch 推理
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                    padder = InputPadder(imgs0.shape, divis_by=32, force_square=False)
                    imgs0, imgs1 = padder.pad(imgs0, imgs1)
                    with torch.cuda.amp.autocast(True):
                        disp = model(imgs0, imgs1, iters=args.valid_iters, test_mode=True)
                    disp = padder.unpad(disp.float()).cpu().numpy()

                t1 = time.time()
                print(f"Batch {idx//B}: time {t1-t0:.3f}s")

                # 保存深度图
                for b, left_file in enumerate(batch_left_files):
                    H, W = sizes[b]
                    disp_b = disp[b].reshape(H, W)
                    depth = disparity_to_depth(disp_b, fx, baseline, scale)
                    depth_mm = (depth * 1000.0).clip(0, args.z_far*1000).astype(np.uint16)
                    depth_filename = os.path.splitext(left_file)[0] + '.png'
                    depth_path = os.path.join(args.out_dir, depth_filename)
                    cv2.imwrite(depth_path, depth_mm)
