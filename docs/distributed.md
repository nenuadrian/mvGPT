# Training on several GPUs

Every model so far trained for ten minutes on a laptop. `pretrain_ddp.py` trains one about six times
bigger, GPT-2 small's shape, on four GPUs rented by the hour, with data parallelism: each GPU holds a
whole copy of the model and trains on batches of its own, and after each backward pass, the GPUs
average their gradients, so all the copies take the same step and stay the same.

## In the code

PyTorch's `torchrun` starts a process per GPU, and tells each its rank, its GPU on the machine, and
how many there are. `pretrain_ddp.py` then differs from `pretrain.py` in a few places:

- **The process group:** NCCL, NVIDIA's library for moving data between GPUs, or gloo on a CPU, to try
  it out on a laptop.
- **The batches:** each rank seeds its random batches with its rank. With one seed, they'd all
  train on the same data, and averaging would change nothing.
- **DistributedDataParallel** starts all the copies from rank 0's weights, and averages the
  gradients across the GPUs during the backward pass, in buckets, while the gradients of the
  layers before are still being computed.
- **The clock:** rank 0's elapsed time is broadcast each step, so all the ranks use the same learning
  rate and stop at the same step. A rank that stopped one step early would leave the others waiting
  for its gradients.
- **Rank 0** alone evaluates, writes to TensorBoard, and saves, at each evaluation, as a rented
  machine can go away.

On NVIDIA GPUs, the forward pass runs in bfloat16, and `--compile` compiles the model with
`torch.compile`. The model is 12 blocks of 12 heads, an embedding size of 768 and a context of 512
tokens, with rotary embeddings and the [BPE tokenizer](bpe.md): 91,347,456 parameters.

`data.py --full` downloads all of TinyStories' training stories, 2.2 GB, and tokenizes them with the
merges learned on the laptop, so the tokens, and the held-out stories, are the ones the laptop's
models were measured on. It splits the text into pieces that end after a story, and encodes them on
every core. `remote.sh HOST` gets a rented machine ready: the code at `HEAD`, the merges, the C++
tokenizer built, and the data.

```bash
./remote.sh vast        # an ssh host
ssh vast 'cd mvGPT && torchrun --standalone --nproc_per_node 4 pretrain_ddp.py 30 --full --compile --batch 96'
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 pretrain_ddp.py 1 --layers 2 --heads 2 --embd 64 --block 64 --batch 8   # on a CPU
```

## How it scales

Four RTX 4090s, rented on [Vast.ai](https://cloud.vast.ai/?ref_id=746308) (a referral link), with no
NVLink between them: the gradients go through the CPU's PCIe bridges. Tokens per second, after the
first steps, which compile:

| GPUs | Batch per GPU | Tokens per second | Against one GPU |
| --- | --- | --- | --- |
| 1 | 48 | 135,200 | |
| 2 | 48 | 230,000 | 1.70× |
| 4 | 48 | 391,600 | 2.90× |
| 1 | 96 | 140,500 | |
| 4 | 96 | 473,700 | 3.37× |

Each step, the GPUs average 365 MB of gradients, 91 million numbers in 32 bits. With twice the
batch on each GPU, a step computes twice as much for the same averaging, so four GPUs get closer
to four times one: 84% of it, against 72%. `torch.compile` gave 383,000 tokens per second against
285,000 without it, on four GPUs at a batch of 48, and used 10.4 GB per GPU instead of 16.2.

## Thirty minutes on four GPUs

```bash
ssh vast 'cd mvGPT; nohup torchrun --standalone --nproc_per_node 4 pretrain_ddp.py 30 --full --compile --batch 96 > train.log 2>&1 < /dev/null & echo $! > train.pid'
ssh vast 'cd mvGPT && tail -f --pid $(cat train.pid) train.log'
scp vast:mvGPT/checkpoints/pretrain-ddp/model.pt checkpoints/pretrain-ddp/
```

The run goes on in the background, so it doesn't stop if the connection drops. In 30.1 minutes, it
took 4,096 steps of 196,608 tokens, 805 million tokens, about one and a half times the training
stories, at 446,000 tokens per second, evaluations included. It passed the laptop model's final
held-out loss, 2.12, within its first 250 steps, about two minutes in. The machine, rented for
about an hour in all, cost $2.62.

| | Laptop, 10 minutes | Four RTX 4090s, 30 minutes |
| --- | --- | --- |
| Parameters | 13,793,280 | 91,347,456 |
| Context | 256 tokens | 512 tokens |
| Tokens trained on | 15.2 million | 805 million |
| Held-out loss, per token | 2.12 | 1.07 |
| Bits per character (`bits.py`) | 0.763 | 0.386 |

Both use rotary embeddings and the same BPE tokenizer, and are measured on the same held-out
stories, so the loss per token compares too. From the same seed, the laptop's model writes:

> Once upon a time, there was a little boy named Tim. Tim liked to play with his toys and
> pretend he was a fast. One day, he found a shiny thing in his yard. He thought it would be
> fun to play with it.

and the one trained on four GPUs:

> Once upon a time, there was a little boy named Tim. Tim liked to play with his toys and
> make crazy messes. One day, Tim's mom told him to clean up his room. She said, "Tim,
> please clean up your toys and use soap to clean your room."

It's on Hugging Face as the `ep012-pretrain` revision of
[nenuadrian/mvGPT](https://huggingface.co/nenuadrian/mvGPT), saved as a GPT-NeoX by `hub.py`.
