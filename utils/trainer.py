import os
import torch
import argparse
import torch.nn as nn

class Trainer:
    def __init__(self, args):
        self.args = args
        self.checkpoint = _CheckPoint(self.args.model_path, self.args.ct, self.args.model_name)

    def train(self, model, optimizer, scheduler, train_dataloader, validation_dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Initialize early stopping
        patience = 10
        epoch_worse = 0
        best_val_loss = float("Inf")
        dice_loss = DiceLoss(self.args.num_classes)
        ce_loss = nn.CrossEntropyLoss()
        print(f"Start Training {self.args.model_name}...")
        start_epoch, loss_train, loss_val = self.checkpoint.maybe_load(model, optimizer, scheduler)
        for epoch in range(start_epoch, self.args.num_epochs):
            # Training
            print("\nEPOCH " + str(epoch + 1) + " of " + str(self.args.num_epochs) + "\n")
            model.train()
            train_loss = 0
            for batch in train_dataloader:
                train_image = batch["data"]
                train_target = batch["label"].long()
                train_image = train_image.to(device)
                train_target = train_target.to(device)
                optimizer.zero_grad()
                train_pred = model(train_image)
                batch_loss = 0.5 * (dice_loss(train_pred, train_target) + ce_loss(train_pred, train_target))
                train_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            loss_train.append(train_loss / len(train_dataloader))
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in validation_dataloader:
                    val_image = batch["data"]
                    val_target = batch["label"].long()
                    val_image = val_image.to(device)
                    val_target = val_target.to(device)
                    val_pred = model(val_image)
                    batch_loss = 0.5 * (dice_loss(val_pred, val_target) + ce_loss(val_pred, val_target))
                    val_loss += batch_loss.item()
            loss_val.append(val_loss / len(validation_dataloader))

            # Update learning rate
            scheduler.step()

            # Progress check
            if epoch % 1 == 0:
                print(f"training loss: {loss_train[-1]},"
                      f"\nvalidation loss: {loss_val[-1]}")
            # Early stop when validation loss stop decreasing
            if loss_val[-1] < best_val_loss:
                best_val_loss = loss_val[-1]
                epoch_worse = 0
                self.checkpoint.save(model, optimizer, scheduler, epoch, loss_train, loss_val)
            else:
                epoch_worse += 1
            if epoch_worse >= patience:
                print(f"Early stop due to validation loss not decrease for {epoch_worse} epochs")
                break

        return loss_train, loss_val


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(target, num_classes=self.n_classes)
        target = target.permute(0, 3, 1, 2)
        loss = 0.0
        smooth = 1e-5
        for i in range(0, self.n_classes):
            intersect = torch.sum(pred[:, i] * target[:, i])
            y_sum = torch.sum(target[:, i] * target[:, i])
            z_sum = torch.sum(pred[:, i] * pred[:, i])
            dice = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            loss += dice
        return loss / self.n_classes


class _CheckPoint:
    def __init__(self, dir, ct, model_name):
        self.checkpoint = os.path.join(dir, model_name + "_checkpoint.pt")
        self.ct_train = ct

    def maybe_load(self, model, optim, lr_scheduler):
        if os.path.exists(self.checkpoint):
            print(f"Load checkpoint from {self.checkpoint}")
            checkpoint = torch.load(self.checkpoint)
            model.load_state_dict(checkpoint["model_state"])
            if self.ct_train:
                optim.load_state_dict(checkpoint["optimizer_state"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
                print(f"Continue training from {checkpoint['epoch']}")
                return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["val_loss"]
            else:
                return 0, [], []
        else:
            print("No checkpoint found to load.")
            return 0, [], []

    def save(self, model, optim, lr_scheduler, epoch, loss_train, loss_val):
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            "train_loss": loss_train,
            "val_loss": loss_val
        }, self.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--model_path", type=str, default=".\checkpoints")
    parser.add_argument("--ct", type=bool, default=False)
    parser.add_argument('--model_name', type=str, default="unet")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=5)
    args = parser.parse_args([])
