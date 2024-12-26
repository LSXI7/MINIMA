import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
from einops.einops import rearrange
import torch.nn.functional as F


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    elif dataset_name == 'vistir':
        thr = 5e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=300, path=None,svg=False):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    mkpts0 = mkpts0.astype(np.int32)
    mkpts1 = mkpts1.astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # if kpts0 is not None:
    #     assert kpts1 is not None
    #     axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
    #     axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]

        # axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        # axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=18, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        if not svg:
            plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            svg_path = path.replace('.jpg', '.pdf')
            from matplotlib import rcParams
            rcParams['svg.fonttype'] = 'none'  # 'none' 
            plt.savefig(svg_path, bbox_inches='tight', pad_inches=0)
    else:
        return fig

def make_matching_figure2(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=300, path=None,svg=False):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    mkpts0 = mkpts0.astype(np.int32)
    mkpts1 = mkpts1.astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # if kpts0 is not None:
    #     assert kpts1 is not None
    #     axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
    #     axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)
    color = color / 255.0
    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=tuple(color[i]), linewidth=1)
                                        for i in range(len(mkpts0))]

        # axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        # axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=18, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        if not svg:
            plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            svg_path = path.replace('.jpg', '.pdf')
            from matplotlib import rcParams
            rcParams['svg.fonttype'] = 'none'  # 'none' 
            plt.savefig(svg_path, bbox_inches='tight', pad_inches=0)

    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', ret_dict=None):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}']
    if ret_dict is not None:
        text += [f"t_err: {ret_dict['metrics']['t_errs'][b_id]:.2f}",
                f"R_err: {ret_dict['metrics']['R_errs'][b_id]:.2f}"]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation', ret_dict=None):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_XoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, ret_dict=ret_dict)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures

def make_mae_figures(data):
    """ Make mae figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_XoFTR_Pretrain.
    Returns:
        figures (List[plt.figure])
    """

    scale = data['hw0_i'][0] // data['hw0_f'][0]
    W_f = data["W_f"]

    pred0, pred1 = data["pred0"], data["pred1"]
    target0, target1 = data["target0"], data["target1"]

    # replace masked regions with predictions
    target0[data['b_ids'][data["ids_image0"]], data['i_ids'][data["ids_image0"]]] = pred0[data["ids_image0"]]
    target1[data['b_ids'][data["ids_image1"]], data['j_ids'][data["ids_image1"]]] = pred1[data["ids_image1"]]

    # remove excess parts, since the 10x10 windows have overlaping regions
    target0 = rearrange(target0, 'n l (h w) (p q c) -> n c (h p) (w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)
    target1 = rearrange(target1, 'n l (h w) (p q c) -> n c (h p) (w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)
    # target0[:,:,-scale:,:] = 0.0
    # target0[:,:,:,-scale:] = 0.0
    # target1[:,:,-scale:,:] = 0.0
    # target1[:,:,:,-scale:] = 0.0
    gap = scale //2
    target0[:,:,-gap:,:] = 0.0
    target0[:,:,:,-gap:] = 0.0
    target1[:,:,-gap:,:] = 0.0
    target1[:,:,:,-gap:] = 0.0
    target0[:,:,:gap,:] = 0.0
    target0[:,:,:,:gap] = 0.0
    target1[:,:,:gap,:] = 0.0
    target1[:,:,:,:gap] = 0.0
    target0 = rearrange(target0, 'n c (h p) (w q) l -> n (c h p w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)
    target1 = rearrange(target1, 'n c (h p) (w q) l -> n (c h p w q) l', h=W_f, w=W_f, p=scale, q=scale, c=1)

    # windows to image 
    kernel_size = [int(W_f*scale), int(W_f*scale)]
    padding = kernel_size[0]//2 -1 if kernel_size[0] % 2 == 0 else kernel_size[0]//2
    stride = data['hw0_i'][0] // data['hw0_c'][0]
    target0 = F.fold(target0, output_size=data["image0"].shape[2:], kernel_size=kernel_size, stride=stride, padding=padding)
    target1 = F.fold(target1, output_size=data["image1"].shape[2:], kernel_size=kernel_size, stride=stride, padding=padding)

    # add mean and std of original image for visualization
    if ("image0_norm" in data) and ("image1_norm" in data):
        target0 = target0 * data["image0_std"] + data["image0_mean"]
        target1 = target1 * data["image1_std"] + data["image1_mean"]
        masked_image0 = data["masked_image0"] * data["image0_std"].to("cpu") + data["image0_mean"].to("cpu")
        masked_image1 = data["masked_image1"] * data["image1_std"].to("cpu") + data["image1_mean"].to("cpu")
    else:
        masked_image0 = data["masked_image0"]
        masked_image1 = data["masked_image1"]

    figures = []
    # Create a list of these tensors
    image_groups = [[data["image0"], masked_image0, target0],
                     [data["image1"], masked_image1, target1]]

    # Iterate through the batches
    for batch_idx in range(image_groups[0][0].shape[0]):  # Assuming batch dimension is the first dimension
        fig, axs = plt.subplots(2, 3, figsize=(9, 6))
        for i, image_tensors in enumerate(image_groups):
            for j, img_tensor in enumerate(image_tensors):
                img = img_tensor[batch_idx, 0, :, :].detach().cpu().numpy()  # Get the image data as a NumPy array
                axs[i,j].imshow(img, cmap='gray', vmin=0, vmax=1)  # Display the image in a subplot with correct colormap
                axs[i,j].axis('off')  # Turn off axis labels
        fig.tight_layout()
        figures.append(fig)
    return figures

def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
