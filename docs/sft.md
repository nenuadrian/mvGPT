# Supervised fine-tuning

A pretrained model continues text. Give it an instruction and it carries on writing whatever looks
likely after it, which is rarely what the instruction asked for. Supervised fine-tuning (SFT) trains
it further on pairs of an instruction and the answer we want, so that it learns to answer. Here the
instructions ask for a story with given words, features or a summary, and the answers are the
stories.

## The data

[TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) has the
stories of TinyStories, each with some of these fields:

```text
Features: Dialogue
Words: quit, oak, gloomy
Summary: Sara and Ben were playing in the park, but Sara wanted to go home ...
Story:

Sara and Ben were playing in the park. They liked to climb the big oak tree ...
<|endoftext|>
```

The fields come in any order, and in almost a third of the documents the summary comes after the
story. [`sft.py`](https://github.com/nenuadrian/mvGPT/blob/main/sft.py) parses each document into
its fields and writes the instruction in one order, features, words, a random sentence and the
summary, ending with `Story:`. The story follows as the completion, and the end-of-text token after
it, so the model also learns where a story should end.

## Learning the story, not the instruction

Each example is the instruction's tokens followed by the story's. In pretraining, every position
is a target. In SFT, the targets over the instruction are set to `-1`, and `F.cross_entropy` skips
them (`ignore_index=-1`), so the loss covers the story alone: the model learns to write stories for
instructions, not to write instructions.

| Input | `Words` | `:` | `dog` | ... | `Story` | `:` | `Once` | `upon` | ... | `.` | `[PAD]` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Target | -1 | -1 | -1 | ... | -1 | `Once` | `upon` | `a` | ... | `<\|endoftext\|>` | -1 |

The last position of the instruction, the `:` after `Story`, predicts the first word of the story,
so that's where the targets start, and the story's last token predicts the end of the text.
Examples have different lengths, so each batch is padded to its longest example with `[PAD]`, and
the padding's targets are `-1` too.

## The training loop

It starts from the pretrained checkpoint, with a learning rate of 2e-4, a third of pretraining's
peak: the model already writes stories, and a smaller step adjusts that without overwriting it.
Otherwise the loop is pretraining's: AdamW, gradient clipping, an evaluation every 250 steps with a
story written for one instruction, and a time budget.

```bash
python sft.py 10
python sample.py checkpoints/sft/model.pt "Words: dog, ball, happy. Story:"
```

## Evaluation

[`evaluate.py`](https://github.com/nenuadrian/mvGPT/blob/main/evaluate.py) compares checkpoints on
held-out data. Along with the story metrics from pretraining, it now has a story written for 50
held-out instructions that ask for words, and reports:

- **instruction_loss**, the loss on the stories of 500 held-out instructions;
- **words_used**, how many of the requested words the stories use, counting plurals and the
  `-d`, `-ed` and `-ing` forms;
- **stopped**, how many stories end with the end-of-text token within 200 tokens;
- **story_tokens**, the stories' average length.

```bash
python evaluate.py checkpoints/pretrain/model.pt checkpoints/sft/model.pt --out results
```

## What to expect

Ten minutes of SFT on an M4 Pro is about 1,170 steps. Compared with the pretrained model:

| | Pretrained | Fine-tuned |
| --- | --- | --- |
| Requested words used | about 3% | about 27% |
| Loss on the instructions' stories | 2.60 | 2.13 |
| Loss on plain stories | 2.25 | 2.38 |
| Average story length, in tokens | 85 | 178 |
| Stories that end within 200 tokens | 92% | 48% |

The stories use about ten times as many of the words they're asked for. They're also about twice as
long, closer to the instructions' stories, so fewer of them end within 200 tokens: the pretrained
model carries on from the middle of a story and soon finishes it, while the fine-tuned one writes a
whole story. Stories longer than the context window also lose their end-of-text token when they're
cut to it, so the model sees fewer endings. The loss on plain stories rises a little, as the model
moves towards the instructions' stories, which come from TinyStories' first version, partly written
by GPT-3.5, while the validation stories come from the second, written by GPT-4.
