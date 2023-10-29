# GILD: Graph-Informed Layout and Design for Inspiration under Constraints

To start, you should first train the [GUILGET method](https://github.com/dysoxor/GUILGET)
The instructions are already written, however some paths must be redefined

In the following we give the instructions to train and infer results from the second step. But before that let's setup your environment:

```shell
conda env create -f environment.yaml
conda activate control
```

If you want to try a pre-trained model, here are [some weights](https://huggingface.co/datasets/iasobolev/controlnet/tree/main/version_4) you have to put in `lightning_log/version_x/` and go directly to the `Step 5 - generate!` below



## Step 1 - Get the dataset ready

Upload the clay dataset as follows:

    ControlNet/training/clay/prompt.json
    ControlNet/training/clay/source/X.png
    ControlNet/training/clay/target/X.png

In the folder "clay/source", you will store the color-coded layouts. For getting the color-coded layouts from bounding box layout, use the [generate_clay_gt.py](https://huggingface.co/datasets/iasobolev/guilget/tree/main). It is also important to have a square shaped input, for that you can use the [rescale.py](https://huggingface.co/datasets/iasobolev/guilget/tree/main) for reducing the size, then use [borders.py](https://huggingface.co/datasets/iasobolev/guilget/tree/main) to add grey default borders that will allow to have a squared image without deformation of the input.

In the folder "clay/target", you will store the design associated to each layout. You might need to use [rescale.py](https://huggingface.co/datasets/iasobolev/guilget/tree/main) and [borders.py](https://huggingface.co/datasets/iasobolev/guilget/tree/main) here as well

In the "clay/prompt.json", you will have their filenames and prompts. Each prompt is a default one "High quality, detailed, and professional app interface". More precisely, here is the expected format for the first two training data:

```json
{"source": "source/68068.png", "target": "target/68068.png", "prompt": "High quality, detailed, and professional app interface"}
{"source": "source/59725.png", "target": "target/59725.png", "prompt": "High quality, detailed, and professional app interface"}
...
```

## Step 2 - Load the dataset

Then you need to write a simple script to read this dataset for pytorch. (In fact we have written it for you in "tutorial_dataset.py".)

```python
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/clay/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/clay/' + source_filename)
        target = cv2.imread('./training/clay/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

```

This will make your dataset into an array-like object in python. You can test this dataset simply by accessing the array, like this

```python
from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

```

The outputs of this simple test on my machine are 

    47626
    High quality, detailed, and professional app interface
    (512, 512, 3)
    (512, 512, 3)

And this code is in "tutorial_dataset_test.py".

In this way, the dataset is an array-like object with 47626 items. Each item is a dict with three entry "jpg", "txt", and "hint". The "jpg" is the target image, the "hint" is the control image, and the "txt" is the prompt. 

Do not ask us why we use these three names - this is related to the dark history of a library called LDM.

## Step 3 - What SD model do you want to control?

Then you need to decide which Stable Diffusion Model you want to control. In this example, we will just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

(Or ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) if you are using SD2.)

Then you need to attach a control net to the SD model.

Note that all weights inside the ControlNet are also copied from SD so that no layer is trained from scratch, and you are still finetuning the entire model.

We provide a simple script for you to achieve this easily. If your SD filename is "./models/v1-5-pruned.ckpt" and you want the script to save the processed model (SD+ControlNet) at location "./models/control_sd15_ini.ckpt", you can just run:

    python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt

Or if you are using SD2:

    python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt

You may also use other filenames as long as the command is "python tool_add_control.py input_path output_path".


## Step 4 - Train!

Happy! We finally come to the most exciting part: training!

The training code in "tutorial_train.py" is actually surprisingly simple:

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)

```
(or "tutorial_train_sd21.py" if you are using SD2)

Thanks to our organized dataset pytorch object and the power of pytorch_lightning, the entire code is just super short.

Now, you may take a look at [Pytorch Lightning Official DOC](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#trainer) to find out how to enable many useful features like gradient accumulation, multiple GPU training, accelerated dataset loading, flexible checkpoint saving, etc. All these only need about one line of code. Great!

Note that if you find OOM, perhaps you need to enable [Low VRAM mode](low_vram.md), and perhaps you also need to use smaller batch size and gradient accumulation. Or you may also want to use some “advanced” tricks like sliced attention or xformers. For example:

```python
# Configs
batch_size = 1

# Misc
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower
```

Note that training with 8 GB laptop GPU is challenging. We will need some GPU memory optimization at least as good as automatic1111’s UI. This may require expert modifications to the code.

## Other options

Beyond standard things, we also provide two important parameters "sd_locked" and "only_mid_control" that you need to know.

### only_mid_control

By default, only_mid_control is False.

This can be helpful when your computation power is limited and want to speed up the training, or when you want to facilitate the "global" context learning. Note that sometimes you may pause training, set it to True, resume training, and pause again, and set it again, and resume again. 

If your computation device is good, perhaps you do not need this. But I also know some artists are willing to train a model on their laptop for a month - in that case, perhaps this option can be useful.

### sd_locked

By default, sd_locked is True.

This will unlock some layers in SD and you will train them as a whole.

This option is DANGEROUS! If your dataset is not good enough, this may downgrade the capability of your SD model.

However, this option is also very useful when you are training on images with some specific style, or when you are training with special datasets (like medical dataset with X-ray images or geographic datasets with lots of Google Maps). You can understand this as simultaneously training the ControlNet and something like a DreamBooth.

Also, if your dataset is large, you may want to end the training with a few thousands of steps with those layer unlocked. This usually improve the "problem-specific" solutions a little. You may try it yourself to feel the difference.

Also, if you unlock some original layers, you may want a lower learning rate, like 2e-6.

## More Consideration: Sudden Converge Phenomenon and Gradient Accumulation

Because we use zero convolutions, the SD should always be able to predict meaningful images. (If it cannot, the training has already failed.)

You will always find that at some iterations, the model "suddenly" be able to fit some training conditions. This means that you will get a basically usable model at about 3k to 7k steps (future training will improve it, but that model after the first "sudden converge" should be basically functional).

Note that 3k to 7k steps is not very large, and you should consider larger batch size rather than more training steps. If you can observe the "sudden converge" at 3k step using batch size 4, then, rather than train it with 300k further steps, a better idea is to use 100× gradient accumulation to re-train that 3k steps with 100× batch size. Note that perhaps we should not do this *too* extremely (perhaps 100x accumulation is too extreme), but you should consider that, since "sudden converge" will *always* happen at that certain point, getting a better converge is more important.

Because that "sudden converge" always happens, lets say "sudden converge" will happen at 3k step and our money can optimize 90k step, then we have two options: (1) train 3k steps, sudden converge, then train 87k steps. (2) 30x gradient accumulation, train 3k steps (90k real computation steps), then sudden converge.

In my experiments, (2) is usually better than (1). However, in real cases, perhaps you may need to balance the steps before and after the "sudden converge" on your own to find a balance. The training after "sudden converge" is also important.

But usually, if your logic batch size is already bigger than 256, then further extending the batch size is not very meaningful. In that case, perhaps a better idea is to train more steps. I tried some "common" logic batch size at 64 or 96 or 128 (by gradient accumulation), it seems that many complicated conditions can be solved very well already.

## Step 5 - Generate!

you have now your model, to try it change the path to your weights in `gradio_seg2image.py`, then run it

```shell
python gradio_seg2image.py
```

you will have then access to an interface running locally where you can try the model.

If you want run experiments with many designs, the manual generation with `gradio_seg2image.py` might be very time consuming, for this purpose I made the `generate.py` that can automatically generate designs for each layout of a directory. There are many parameters that you can play with depending on the purpose of your experiment. *Don't forget to change the path to the weight* 

```shell
python generate.py
```

## Step 6 - Evaluate

For the evaluation of the generated designs we used the FID and diversity score which are the most common metrics in the field. You can find the codes to do it in [fid.py](https://huggingface.co/datasets/iasobolev/controlnet/tree/main/version_4) and [diversity.py](https://huggingface.co/datasets/iasobolev/controlnet/tree/main/version_4). Before evaluating, it would be better to crop the generated designs to remove gray borders, [crop.py](https://huggingface.co/datasets/iasobolev/controlnet/tree/main/version_4) allows to do that. 

### Acknowledgements
This code borrows heavily from [ControlNet](https://github.com/lllyasviel/ControlNet) repository. Many thanks.
