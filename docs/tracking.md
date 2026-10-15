# Watching training

`pretrain.py` and `sft.py` log each run to [TensorBoard](https://www.tensorflow.org/tensorboard),
through PyTorch's `SummaryWriter`. Each run gets a folder of its own under `runs/`, named after the
run and when it started, such as `runs/sft-20261015-101500`. A new run doesn't overwrite the last
one, so runs can be compared side by side.

| Tag | Logged | What it is |
| --- | --- | --- |
| `loss/batch` | every 50 steps | the loss on that step's training batch |
| `loss/train`, `loss/valid` | every 250 steps | the loss averaged over 20 batches of training and held-out data |
| `lr` | every 50 steps | the learning rate (pretraining; fine-tuning keeps it at 2e-4) |
| `tokens_per_s` | every 50 steps | how fast it trains (pretraining) |
| `sample` | every 250 steps | the text the model writes for the run's prompt |

```bash
tensorboard --logdir runs    # http://localhost:6006
```

TensorBoard reads the folder as the runs write to it, so a run can be watched while it trains. Its
Scalars tab has a chart per tag, and its Text tab has the samples, by step.

## Reading the logs in Python

The logs are TensorBoard's event files, and TensorBoard's own reader, `EventAccumulator`, loads them
in Python. [`plot.py`](https://github.com/nenuadrian/mvGPT/blob/main/plot.py) uses it to print each
tag and the samples, and plots the loss and the learning rate with matplotlib:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

events = EventAccumulator("runs/sft-20261015-101500")
events.Reload()
for point in events.Scalars("loss/valid"):
    print(point.step, point.value)
```

```bash
python plot.py runs/sft-20261015-101500    # writes results/sft-20261015-101500.png
```

Both kinds of loss go on one chart: the training batches are noisy, while the held-out loss is an
average and changes slowly, and the gap between them shows if the model memorises its training data.
