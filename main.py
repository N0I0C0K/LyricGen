import glob

import numpy as np
import torch
import torch.nn.functional as fc
from rich.progress import Progress

from model import SongNet

# from pprint import pprint

try:
    from rich.console import Console

    pprint = Console().print
except:
    pprint = print
from dataset import DataLoader, SongData, random_split

debug = False


def accuracy(out: torch.Tensor, target: torch.Tensor, raw_size: int):
    out = out.reshape(-1, raw_size)
    target = target.flatten()

    a = out.topk(1).indices.flatten()
    return a.eq(target).sum().item() / len(a)


class SongGen:
    def __init__(self) -> None:
        self.seq_len = 48
        self.batch_size = 64
        self.embedding_size = 128
        self.lstm_hidden_size = 512
        self.lr = 0.01

        self.device = torch.device("cuda:0")
        self.dataset = SongData(
            self.seq_len,
            lines=2000 if debug else -1,
        )
        data_len = len(self.dataset)
        train_data, test_data = random_split(self.dataset, [data_len - 1000, 1000])
        self.train_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, self.batch_size, shuffle=True)

        # if True:
        #     t = 1
        #     for i, data in enumerate(self.train_loader):
        #         if t == 0:
        #             break
        #         input_batch, out_put = data
        #         pprint(input_batch)
        #         pprint(input_batch.shape)
        #         t -= 1
        #         # first = input_batch[0]
        #         # pprint("".join([dataset.index2word[x.item()] for x in first]))

        self.raw_size = len(self.dataset.word2index)
        self.model = SongNet(
            self.raw_size, self.embedding_size, self.lstm_hidden_size
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

        checkpoint_files = glob.glob("checkpoint-*.pth")

        self.epoch = 0
        self.check_file = None
        if (
            checkpoint_files
            and input("Enter y to load:%s >" % checkpoint_files[-1]) == "y"
        ):
            self.load_checkpoint(checkpoint_files[-1])
            self.check_file = checkpoint_files[-1]
        self.process: Progress = Progress()

    def generate(self, start_phrases: str):
        self.model.eval()
        words = start_phrases.split("/")

        hidden = None

        def next_word(word: str, noise: bool = False):
            nonlocal hidden
            input_index = self.dataset.word2index[word]
            # if not noise:
            input_ = torch.Tensor([[input_index]]).long().to(self.device)
            # else:
            #     input_ = torch.Tensor([[input_index]]).float()
            #     nois = torch.rand_like(input_) * 0.01
            #     input_ = (input_ + nois).to(self.device)

            out_, hidden = self.model(input_, hidden, noise)
            top_word_idx = out_[0].topk(1).indices.item()
            return self.dataset.index2word[top_word_idx]

        res: list[str] = []
        while words:
            word = words.pop(0)
            line_s = []
            last_word = ""
            for si in word:
                line_s.append(si)
                next_word(si, True)
                last_word = si
            while last_word != "/":
                t = next_word(last_word, True)
                line_s.append(t)
                last_word = t
            res.append("".join(line_s))
        return "".join(res)

    def generate_all(self):
        self.model.eval()
        hidden = None
        last_word = "<SOS>"
        res = []

        hidden = None

        def next_word(word: str, noise: bool = False):
            nonlocal hidden
            input_index = self.dataset.word2index[word]
            # if not noise:
            input_ = torch.Tensor([[input_index]]).long().to(self.device)
            # else:
            #     input_ = torch.Tensor([[input_index]]).float()
            #     nois = torch.rand_like(input_) * 0.01
            #     input_ = (input_ + nois).to(self.device)

            out_, hidden = self.model(input_, hidden, noise)
            top_word_idx = out_[0].topk(1).indices.item()
            return self.dataset.index2word[top_word_idx]

        while last_word != "<EOS>" and len(res) <= 128:
            t = next_word(last_word, True)
            last_word = t
            res.append(t)

        return "".join(res)

    def train_step(self):
        self.model.train()
        task = self.process.add_task("Training...", total=len(self.train_loader))
        loss_total = 0
        for i, (input_, target_) in enumerate(self.train_loader):
            input_, target_ = input_.to(self.device), target_.to(self.device)
            out: torch.Tensor = self.model(input_)[0]
            loss = fc.cross_entropy(out.reshape(-1, self.raw_size), target_.flatten())
            loss_total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # pprint(f"{input_.shape} {out.shape} {target_.shape}")
            # pprint(
            #     f"Training: Epoch={self.epoch} Batch={i}/{len(self.train_loader)} Loss={loss.item():.4f}"
            # )
            self.process.update(
                task,
                advance=1,
                description=f"Training: Epoch={self.epoch} Batch={i}/{len(self.train_loader)} Loss={loss.item():.4f}",
            )
        self.optimizer_scheduler.step()
        self.process.stop_task(task)
        self.process.remove_task(task)
        self.process.print(
            f"LR={self.optimizer_scheduler.get_last_lr()}  Train AvgLoss={loss_total/len(self.train_loader):.4f}"
        )

    def set_lr(self, new_lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_lr(self):
        res = []
        for param_group in self.optimizer.param_groups:
            res.append(str(param_group["lr"]))
        return ",".join(res)

    def evaluation_step(self):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for input_, target_ in self.test_loader:
                input_, target_ = input_.to(self.device), target_.to(self.device)
                out: torch.Tensor = self.model(input_)[0]

                loss = fc.cross_entropy(
                    out.reshape(-1, self.raw_size), target_.flatten()
                )
                epoch_loss += loss.item()
                epoch_acc += accuracy(out, target_, self.raw_size)

        epoch_acc /= len(self.test_loader)
        epoch_loss /= len(self.test_loader)

        self.process.print(
            f"Validation: Epoch={self.epoch} Average Loss={epoch_loss:.4f} Average Accuracy={epoch_acc:.4f}"
        )

    def start_training(self, epochs: int = 128):
        check_file = self.check_file
        self.process.start()
        tran_task = self.process.add_task("Training...", total=epochs)

        # adjust lr manual
        # self.set_lr(0.005)

        epochs += self.epoch
        if check_file is None:
            import time

            check_file = f'checkpoint-{time.strftime("%y%m%d-%H%M%S")}.pth'

        while self.epoch < epochs:
            self.process.update(tran_task, description=f"LR={self.get_lr()}")
            self.train_step()
            self.evaluation_step()
            self.epoch += 1
            self.save_checkpoint(check_file)
            self.process.print(self.generate("你/我/你/我"))
            self.process.update(tran_task, advance=1)

        self.process.stop()

    def save_checkpoint(self, file_name: str | None = None):
        if file_name is None:
            import time

            file_name = f'checkpoint-{time.strftime("%y%m%d-%H%M%S")}.pth'
        with open(f"./{file_name}", "wb") as file:
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "optimizer_scheduler_state_dict": self.optimizer_scheduler.state_dict(),
                },
                file,
            )

    def load_checkpoint(self, file: str):
        ckpt = torch.load(file)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.optimizer_scheduler.load_state_dict(ckpt["optimizer_scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        pprint(f"load {file}, epoch: {self.epoch}")


def main():
    import time

    torch.manual_seed(time.time())
    model = SongGen()
    model.start_training(36)
    # while (s := input(">")) != "exit":
    #     pprint(model.generate_all())


if __name__ == "__main__":
    main()
