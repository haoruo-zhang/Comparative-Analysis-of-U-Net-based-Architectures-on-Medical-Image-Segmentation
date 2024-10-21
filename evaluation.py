import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_hd95(pred, target):
    # Get the coordinates of the boundary points
    pred_points = torch.nonzero(pred, as_tuple=False)
    target_points = torch.nonzero(target, as_tuple=False)
    if pred_points.numel() == 0 and target_points.numel() == 0:
        return 0
    elif pred_points.numel() == 0 or target_points.numel() == 0:
        return float('inf')
    # Compute pairwise distances between all boundary points
    dists = torch.cdist(pred_points.float(), target_points.float(), p=2)

    # For each point in A, find the minimum distance to a point in B, and vice versa
    min_dists_A_to_B = torch.min(dists, dim=1)[0]
    min_dists_B_to_A = torch.min(dists, dim=0)[0]

    # Compute the 95th percentile of the distances
    hd95_A_to_B = torch.quantile(min_dists_A_to_B, 0.95)
    hd95_B_to_A = torch.quantile(min_dists_B_to_A, 0.95)

    # Return the maximum of the two
    return torch.max(hd95_A_to_B, hd95_B_to_A).item()


def compute_metrics(predictions, targets, num_classes):
    dice_total = 0.0
    accuracy_total = 0.0
    precision_total = 0.0
    hausdorff_dist_total = 0.0

    for i in range(1, num_classes):  # Skip background class (assuming it's class 0)
        pred_i = (predictions == i).float()
        target_i = (targets == i).float()

        TP = (pred_i * target_i).sum().float()
        TN = ((1 - pred_i) * (1 - target_i)).sum().float()
        FP = (pred_i * (1 - target_i)).sum().float()
        FN = ((1 - pred_i) * target_i).sum().float()

        # Special case: No ground truth or predicted positives for this class
        if target_i.sum() == 0 and pred_i.sum() == 0:
            dice = 1.0  # Perfect score since both are empty
            precision = 1.0  # Perfect precision since there's nothing to predict
        else:
            dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
            dice = dice.item()
            precision = TP / (TP + FP + 1e-8)
            precision = precision.item()

        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        hausdorff_dist_95 = compute_hd95(pred_i, target_i)

        dice_total += dice
        accuracy_total += accuracy.item()
        precision_total += precision
        hausdorff_dist_total += hausdorff_dist_95

    num_valid_classes = num_classes - 1
    return (dice_total / num_valid_classes,
            accuracy_total / num_valid_classes,
            precision_total / num_valid_classes,
            hausdorff_dist_total / num_valid_classes)


def evaluation(args, model, test_dataloader):
    checkpoint_path = os.path.join(args.model_path, args.model_name + "_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path)
    loss_train = checkpoint["train_loss"]
    loss_val = checkpoint["val_loss"]
    plt.plot(np.arange(len(loss_train)), loss_train, label="Train Loss")
    plt.plot(np.arange(len(loss_val)), loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("0.5 Dice Loss + 0.5 CE Loss")
    if args.model_name == "unet":
        plt.title("U-Net Loss vs. Epoch")
    elif args.model_name == "transunet":
        plt.title("TransUnet Loss vs. Epoch")
    else:
        plt.title("MambaUnet Loss vs. Epoch")
    plt.legend()
    plt.savefig(os.path.join(args.model_path, args.model_name + ".png"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scores = {
        "index": [],
        "dice": [],
        "accuracy": [],
        "precision": [],
        "hd95": []
    }

    with torch.no_grad():
        for batch in test_dataloader:
            test_idx = batch["index"].item()
            test_image = batch["data"]
            test_label = batch["label"]
            if torch.all(test_label == 0):
                continue
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            test_pred = model(test_image)
            pred_label = torch.argmax(test_pred, 1)
            cur_dice, cur_acc, cur_pre, cur_hd = compute_metrics(pred_label, test_label, args.num_classes)
            scores["index"].append(test_idx)
            scores["dice"].append(cur_dice)
            scores["accuracy"].append(cur_acc)
            scores["precision"].append(cur_pre)
            scores["hd95"].append(cur_hd)

    # Convert scores to a DataFrame and save or display
    df = pd.DataFrame(scores)
    return df

