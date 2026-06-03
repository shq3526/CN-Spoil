"""Dataset loading for CN-Spoil."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpoilExample:
    title: str
    article: str
    image_caption: str
    spoiler: str


class CNSpoilDataset(Dataset):
    """JSONL-based CN-Spoil dataset placeholder loader."""

    def __init__(self, data_path: str, max_samples: int | None = None):
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Place CN-Spoil JSONL files under /data/CN-Spoil/."
            )

        self.examples: List[SpoilExample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.examples.append(
                    SpoilExample(
                        title=row.get("title", ""),
                        article=row.get("article", ""),
                        image_caption=row.get("image_caption", ""),
                        spoiler=row.get("spoiler", ""),
                    )
                )
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.examples[idx]
        return {
            "title": item.title,
            "article": item.article,
            "image_caption": item.image_caption,
            "spoiler": item.spoiler,
        }


def collate_batch(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {k: [item[k] for item in batch] for k in batch[0]}
