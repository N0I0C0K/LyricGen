import numpy
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SongData(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        seq_len: int,
        file: str = "./data/lyrics.txt",
        lines: int = -1,
    ) -> None:
        SOS = 0
        EOS = 1
        self.word2index: dict[str, int] = {"<SOS>": 0, "<EOS>": 1}
        indices: list[int] = []
        num_words: int = 1
        self.seq_len = seq_len

        load_lines = 0
        with open(file, encoding="utf-8") as f:
            while True:
                s = f.readline().strip(" \n\r")
                if len(s) == 0:
                    break
                indices.append(SOS)
                for char in s:
                    if char not in self.word2index:
                        num_words += 1
                        self.word2index[char] = num_words
                    indices.append(self.word2index[char])
                indices.append(EOS)
                load_lines += 1

                if lines != -1 and load_lines >= lines:
                    break

        self.index2word = {v: k for k, v in self.word2index.items()}
        self.data = numpy.array(indices, dtype=numpy.int64)

    def __len__(self) -> int:
        return len(self.data) // self.seq_len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = index * self.seq_len, (index + 1) * self.seq_len
        return (
            torch.as_tensor(self.data[start:end]),
            torch.as_tensor(self.data[start + 1 : end + 1]),
        )
