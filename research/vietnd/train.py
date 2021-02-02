import os
import pickle

import hydra
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, GradsScalarHandler, GradsHistHandler
import transformers

from data import load_data, xnli_process, Dataset
from model import MyEnsemble
from optimizer import RAdam


def run(train_dataloader, test_dataloader, cfg):
    """
    """
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = MyEnsemble(device, cfg)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optim_params = {'lr': cfg.lr_rate, 'eps': cfg.eps}
    if cfg.optimizer == "AdamW":
        optim = transformers.AdamW(optimizer_grouped_parameters, **optim_params)
    elif cfg.optimizer == "RAdam":
        optim = RAdam(optimizer_grouped_parameters, **optim_params)
    elif cfg.optimizer == "Adam":
        optim = torch.optim.Adam(optimizer_grouped_parameters, **optim_params)

    def train_step(engine, batch):
        """
        """
        model.train()
        optim.zero_grad()
        y = batch["label"].to(device)
        y_pred = model(batch["en"], batch["vi"])
        loss = criterion(y_pred, y)
        loss.backward()
        optim.step()

        return loss.item()

    def validation_step(engine, batch):
        """
        """
        model.eval()
        with torch.no_grad():
            y = batch["label"].to(device)
            y_pred = model(batch["en"], batch["vi"])

        return y_pred, y

    def score_function(engine):
        """Use accuracy to determine which model is saved
        """
        return engine.state.metrics['accuracy']

    trainer = Engine(train_step)
    evaluator = Engine(validation_step)

    pbar = tqdm(initial=0, leave=False,
                total=len(train_dataloader),
                desc=f"ITERATION - loss: {0:.2f}")

    # Print accuracy and loss value when validating
    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    # Save model according to computed validation's metric (accuracy)
    to_save = {'model': model}
    handler = Checkpoint(to_save, DiskSaver(cfg.save_model_path, create_dir=True),
                         n_saved=cfg.num_saved,
                         filename_prefix='best',
                         score_function=score_function,
                         score_name="val_acc",
                         global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # Create tensorboard logger
    tb_logger = TensorboardLogger(log_dir=cfg.log_tensorboard_path)

    # Attach the logger to the trainer to log model's gradients norm after each iteration (batch)
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )

    # Attach the logger to the evaluator on the validation dataset after each epoch
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["loss", "accuracy"],
        global_step_transform=global_step_from_engine(trainer)
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.log_interval))
    def log_training_loss(engine):
        # print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
        pbar.desc = f"ITERATION - loss: {engine.state.output:.2f}"
        pbar.update(cfg.log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        # print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(trainer.state.epoch, metrics["accuracy"], metrics["cel"]))
        tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                   .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(trainer.state.epoch, metrics["accuracy"], metrics["cel"]))
        tqdm.write("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}" \
                   .format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))
        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time(engine):
        tqdm.write(f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds")

    trainer.run(train_dataloader, max_epochs=cfg.max_epochs)
    pbar.close()
    tb_logger.close()
    print(f"{cfg.num_saved} model is saved in {cfg.save_model_path}")


@hydra.main(config_name="config.yaml")
def main(cfg):
    """
    """
    if 'xnli' in cfg.use_dataset:
        if 'en_vi_xnli.pkl' not in os.listdir(cfg.dataset.xnli):
            xnli_process(load_data(cfg, version="xnli"))

        with open(cfg.dataset.xnli + '/en_vi_xnli.pkl', 'rb') as f:
            raw = pickle.load(f)
    elif 'snli' in cfg.use_dataset:
        if 'en_vi_snli.pkl' not in os.listdir(cfg.dataset.snli):
            xnli_process(load_data(cfg, version="snli"))

        with open(cfg.dataset.xnli + '/en_vi_snli.pkl', 'rb') as f:
            raw = pickle.load(f)
    else:
        raise NotImplementedError("use_dataset in config.yaml must be 'xnli' or 'snli'")
    train_xnli, test_xnli = train_test_split(pd.DataFrame(raw), test_size=cfg.test_size)
    trainXNLI = Dataset(train_xnli, version=cfg.use_dataset)
    testXNLI = Dataset(test_xnli, version=cfg.use_dataset)
    trainXNLI_dl = DataLoader(trainXNLI, cfg.train_bs, shuffle=True)
    testXNLI_dl = DataLoader(testXNLI, cfg.test_bs, shuffle=False)

    run(trainXNLI_dl, testXNLI_dl, cfg)


if __name__ == "__main__":
    main()
