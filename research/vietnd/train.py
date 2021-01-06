from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from data import get_data, xnli_process, xnliDataset
from model import MyEnsemble


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

model = MyEnsemble(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

max_epoch = 1


def compute_loss(en, vi, true_label):
    """Computes the forward pass and returns the
    cross entropy loss for each sample (en - vi).
    """
    logits = model(en, vi)
    logits = logits.view(-1, logits.size(-1))
    labels = torch.tensor([true_label]).to(device)
    loss = criterion(logits, labels)
    return loss


def training_step(batch):
    """Training phase
    """
    model.train()
    for x, y, z in zip(batch["en"], batch["vi"], batch["label"]):
        optim.zero_grad()
        loss = compute_loss(x, y, z)
        loss.backward()
        optim.step()

    return loss.item()


def eval_step(batch):
    """Evaluate step
    """
    model.eval()
    loss = []
    with torch.no_grad():
        for x, y, z in zip(batch["en"], batch["vi"], batch["label"]):
            loss.append(compute_loss(x, y, z))
    return sum(loss) / len(batch)


def multi_accu(batch):
    """Multiclass accuracy
    """
    logsoft = torch.nn.LogSoftmax(dim=1)
    y_pred = []
    with torch.no_grad():
        for x, y in zip(batch["en"], batch["vi"]):
            out = model(x, y)
            y_pred_softmax = logsoft(out)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred.append(y_pred_tags.item())

    acc = (torch.tensor(y_pred) == batch["label"]).sum() / len(batch["label"])

    return acc.item()


def main(eval=True):
    """
    """
    en_vi_xnli = xnli_process(get_data("xnli"))
    train_xnli, test_xnli = train_test_split(en_vi_xnli, test_size=0.14)
    trainXNLI = xnliDataset(train_xnli)
    testXNLI = xnliDataset(test_xnli)
    trainXNLI_dl = DataLoader(trainXNLI, 32, shuffle=True)
    testXNLI_dl = DataLoader(testXNLI, 32, shuffle=False)

    for epoch in range(1):
        print("Epoch = ", epoch+1)
        for i, batch in enumerate(trainXNLI_dl):
            loss = training_step(batch)
            if i % 5 == 0:
                print(f"Training on batch {i} -- train loss {loss}")
        for i, batch in enumerate(testXNLI_dl):
            print(f"{i} --- {eval_step(batch)} --- {multi_accu(batch):.2%}")

    if eval:
        for index, row in tqdm(test_xnli.iterrows(), total=test_xnli.shape[0]):
            out = model(row["premise"], row["hypothesis"])
            test_xnli.at[index, "pred"] = int(torch.max(out.data, 1)[1].cpu().numpy())


if __name__ == "__main__":
    main()
