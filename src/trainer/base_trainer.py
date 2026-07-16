from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function.
            metrics (dict): training and inference metrics.
            optimizer (Optimizer): optimizer.
            lr_scheduler: learning-rate scheduler.
            config: experiment configuration.
            device (str): computation device.
            dataloaders (dict): train/validation/test dataloaders.
            logger: project logger.
            writer: W&B or CometML writer.
            epoch_len (int | None): number of batches per epoch.
            skip_oom (bool): skip CUDA OOM batches when True.
            batch_transforms (dict | None): transforms applied to batches.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        self.train_dataloader = dataloaders["train"]

        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(
                self.train_dataloader
            )
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            key: dataloader
            for key, dataloader in dataloaders.items()
            if key != "train"
        }

        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get(
            "monitor",
            "off",
        )

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = (
                self.monitor.split()
            )

            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = (
                inf
                if self.mnt_mode == "min"
                else -inf
            )

            self.early_stop = self.cfg_trainer.get(
                "early_stop",
                inf,
            )

            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer
        self.metrics = metrics

        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[
                metric.name
                for metric in self.metrics["train"]
            ],
            writer=self.writer,
        )

        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[
                metric.name
                for metric in self.metrics["inference"]
            ],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH
            / config.trainer.save_dir
            / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = (
                self.checkpoint_dir
                / config.trainer.resume_from
            )
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(
                config.trainer.get("from_pretrained")
            )

    def train(self):
        """
        Run training and always close the experiment tracker.
        """
        try:
            self._train_process()

        except KeyboardInterrupt as error:
            self.logger.info(
                "Saving model on keyboard interrupt"
            )

            self._save_checkpoint(
                self._last_epoch,
                save_best=False,
            )

            raise error

        finally:
            if (
                self.writer is not None
                and hasattr(self.writer, "finish")
            ):
                self.writer.finish()

    def _train_process(self):
        """
        Complete multi-epoch training process.
        """
        not_improved_count = 0

        for epoch in range(
            self.start_epoch,
            self.epochs + 1,
        ):
            self._last_epoch = epoch

            result = self._train_epoch(epoch)

            logs = {
                "epoch": epoch,
            }
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(
                    f"    {key:15s}: {value}"
                )

            # Log all consolidated metrics once per epoch.
            self._log_epoch_metrics(
                epoch=epoch,
                logs=logs,
            )

            (
                best,
                stop_process,
                not_improved_count,
            ) = self._monitor_performance(
                logs,
                not_improved_count,
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(
                    epoch,
                    save_best=best,
                    only_best=True,
                )

            if stop_process:
                break

    def _train_epoch(self, epoch):
        """
        Train the model for one epoch and evaluate it.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()

        self.writer.set_step(
            (epoch - 1) * self.epoch_len
        )
        self.writer.add_scalar("epoch", epoch)

        last_train_metrics = None

        for batch_idx, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.epoch_len,
            )
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )

            except torch.cuda.OutOfMemoryError as error:
                if self.skip_oom:
                    self.logger.warning(
                        "OOM on batch. Skipping batch."
                    )
                    torch.cuda.empty_cache()
                    continue

                raise error

            self.train_metrics.update(
                "grad_norm",
                self._get_grad_norm(),
            )

            if batch_idx % self.log_step == 0:
                global_step = (
                    (epoch - 1) * self.epoch_len
                    + batch_idx
                )

                self.writer.set_step(global_step)

                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        batch["loss"].item(),
                    )
                )

                self.writer.add_scalar(
                    "learning_rate",
                    self.lr_scheduler.get_last_lr()[0],
                )

                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)

                last_train_metrics = (
                    self.train_metrics.result()
                )

                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        # If log_step is larger than epoch_len or all logged batches
        # were skipped, collect the remaining metric state here.
        if last_train_metrics is None:
            last_train_metrics = (
                self.train_metrics.result()
            )

        logs = last_train_metrics

        for (
            part,
            dataloader,
        ) in self.evaluation_dataloaders.items():
            evaluation_logs = self._evaluation_epoch(
                epoch=epoch,
                part=part,
                dataloader=dataloader,
            )

            logs.update(
                {
                    f"{part}_{name}": value
                    for name, value
                    in evaluation_logs.items()
                }
            )

        return logs

    def _evaluation_epoch(
        self,
        epoch,
        part,
        dataloader,
    ):
        """
        Evaluate the model on validation or test data.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        last_batch = None
        last_batch_idx = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )

                last_batch = batch
                last_batch_idx = batch_idx

        self.writer.set_step(
            epoch * self.epoch_len,
            part,
        )

        self._log_scalars(
            self.evaluation_metrics
        )

        if last_batch is not None:
            self._log_batch(
                last_batch_idx,
                last_batch,
                part,
            )

        return self.evaluation_metrics.result()

    def _monitor_performance(
        self,
        logs,
        not_improved_count,
    ):
        """
        Check whether the monitored metric improved.
        """
        best = False
        stop_process = False

        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = (
                        logs[self.mnt_metric]
                        <= self.mnt_best
                    )

                elif self.mnt_mode == "max":
                    improved = (
                        logs[self.mnt_metric]
                        >= self.mnt_best
                    )

                else:
                    improved = False

            except KeyError:
                self.logger.warning(
                    f"Warning: Metric "
                    f"'{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )

                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve "
                    "for {} epochs. Training stops.".format(
                        self.early_stop
                    )
                )
                stop_process = True

        return (
            best,
            stop_process,
            not_improved_count,
        )

    def move_batch_to_device(self, batch):
        """
        Move configured tensors to the selected device.
        """
        for tensor_name in self.cfg_trainer.device_tensors:
            batch[tensor_name] = batch[tensor_name].to(
                self.device
            )

        return batch

    def transform_batch(self, batch):
        """
        Apply configured transforms to a complete batch.
        """
        transform_type = (
            "train"
            if self.is_train
            else "inference"
        )

        transforms = self.batch_transforms.get(
            transform_type
        )

        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[
                    transform_name
                ](
                    batch[transform_name]
                )

        return batch

    def _clip_grad_norm(self):
        """
        Clip gradients when max_grad_norm is configured.
        """
        max_grad_norm = self.config[
            "trainer"
        ].get(
            "max_grad_norm",
            None,
        )

        if max_grad_norm is not None:
            clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculate total gradient norm.
        """
        parameters = self.model.parameters()

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [
            parameter
            for parameter in parameters
            if parameter.grad is not None
        ]

        if not parameters:
            return 0.0

        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        parameter.grad.detach(),
                        norm_type,
                    )
                    for parameter in parameters
                ]
            ),
            norm_type,
        )

        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Format current progress within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"

        if hasattr(
            self.train_dataloader,
            "n_samples",
        ):
            current = (
                batch_idx
                * self.train_dataloader.batch_size
            )
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len

        return base.format(
            current,
            total,
            100.0 * current / total,
        )

    @abstractmethod
    def _log_batch(
        self,
        batch_idx,
        batch,
        mode="train",
    ):
        """
        Log batch-specific objects.

        Must be implemented by the derived Trainer class.
        """
        raise NotImplementedError()

    def _log_epoch_metrics(
        self,
        epoch,
        logs,
    ):
        """
        Log consolidated numeric values once per epoch.

        The epoch number is passed separately and becomes the x-axis
        through WandBWriter.add_epoch_scalars().
        """
        if self.writer is None:
            return

        epoch_logs = {}

        for (
            metric_name,
            metric_value,
        ) in logs.items():
            # Epoch is already supplied separately as epoch_index.
            if metric_name == "epoch":
                continue

            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() != 1:
                    continue

                metric_value = (
                    metric_value
                    .detach()
                    .cpu()
                    .item()
                )

            if isinstance(
                metric_value,
                (int, float),
            ):
                epoch_logs[metric_name] = float(
                    metric_value
                )

        if hasattr(
            self.writer,
            "add_epoch_scalars",
        ):
            self.writer.add_epoch_scalars(
                epoch=epoch,
                scalars=epoch_logs,
            )

    def _log_scalars(
        self,
        metric_tracker: MetricTracker,
    ):
        """
        Log all scalar values from a MetricTracker.
        """
        if self.writer is None:
            return

        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                metric_name,
                metric_tracker.avg(metric_name),
            )

    def _save_checkpoint(
        self,
        epoch,
        save_best=False,
        only_best=False,
    ):
        """
        Save a model checkpoint.
        """
        architecture = type(self.model).__name__

        state = {
            "arch": architecture,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        filename = str(
            self.checkpoint_dir
            / f"checkpoint-epoch{epoch}.pth"
        )

        if not (only_best and save_best):
            torch.save(state, filename)

            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(
                    filename,
                    str(self.checkpoint_dir.parent),
                )

            self.logger.info(
                f"Saving checkpoint: {filename} ..."
            )

        if save_best:
            best_path = str(
                self.checkpoint_dir
                / "model_best.pth"
            )

            torch.save(state, best_path)

            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(
                    best_path,
                    str(self.checkpoint_dir.parent),
                )

            self.logger.info(
                "Saving current best: model_best.pth ..."
            )

    def _resume_checkpoint(self, resume_path):
        """
        Resume model, optimizer and scheduler from a checkpoint.
        """
        resume_path = str(resume_path)

        self.logger.info(
            f"Loading checkpoint: {resume_path} ..."
        )

        checkpoint = torch.load(
            resume_path,
            map_location=self.device,
            weights_only=False,
        )

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if (
            checkpoint["config"]["model"]
            != self.config["model"]
        ):
            self.logger.warning(
                "Warning: Architecture configuration differs "
                "from the checkpoint configuration."
            )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        optimizer_changed = (
            checkpoint["config"]["optimizer"]
            != self.config["optimizer"]
        )

        scheduler_changed = (
            checkpoint["config"]["lr_scheduler"]
            != self.config["lr_scheduler"]
        )

        if optimizer_changed or scheduler_changed:
            self.logger.warning(
                "Warning: Optimizer or scheduler differs from "
                "the checkpoint. Their state was not restored."
            )
        else:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

            self.lr_scheduler.load_state_dict(
                checkpoint["lr_scheduler"]
            )

        self.logger.info(
            "Checkpoint loaded. Resume training "
            f"from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Initialize only model weights from a checkpoint.
        """
        pretrained_path = str(pretrained_path)

        if hasattr(self, "logger"):
            self.logger.info(
                f"Loading model weights from: "
                f"{pretrained_path} ..."
            )
        else:
            print(
                f"Loading model weights from: "
                f"{pretrained_path} ..."
            )

        checkpoint = torch.load(
            pretrained_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(
                checkpoint["state_dict"]
            )
        else:
            self.model.load_state_dict(checkpoint)