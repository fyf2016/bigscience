# Train 1 - 13B - unmodified Megatron gpt2 - baseline（第一个训练模型 - 13B - 未经修改的基于 Megatron 和 GPT-2 架构开发的模型 - 基准模型）

## Task
## 任务

Auto-regressive objective using regular Megatron-LM GPT2 language model

使用常规的 Megatron-LM GPT2 语言模型来实现自回归目标

## Environment（环境）

To launch the environment use [start-tr1-13B](./start-tr1-13B)
要启动环境，请使用 [start-tr1-13B](./start-tr1-13B)。


```
source $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B
```

We are using the following branches specific to this training（我们正在使用以下与此训练相关的分支）:

- `$six_ALL_CCFRWORK/code/tr1-13B/Megatron-DeepSpeed-tr1-13B` a frozen `tr1-13B` branch - can cherry pick from `main` if need be.（`$six_ALL_CCFRWORK/code/tr1-13B/Megatron-DeepSpeed-tr1-13B` 是一个锁定的 tr1-13B 分支 - 如果需要，可以从 main 分支进行挑选（cherry-pick））
- `$six_ALL_CCFRWORK/code/tr1-13B/DeepSpeed-big-science` - a mostly frozen `big-science` branch - under Deepspeed's team control - so it may also require a specific SHA if something gets broken upstream.（`$six_ALL_CCFRWORK/code/tr1-13B/DeepSpeed-big-science` 是一个大部分锁定的 big-science 分支，由 Deepspeed 团队控制，因此如果上游出现问题，可能还需要一个特定的 SHA（提交哈希值））

How the environment was built（环境是如何构建的）：
```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n tr1-13B python=3.8
conda activate tr1-13B
conda install pytorch==1.8.1 torchvision cudatoolkit=10.2 -c pytorch -y
pip install deepspeed
pip install tensorboard

mkdir $six_ALL_CCFRWORK/code/tr1-13B

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/bigscience-workshop/bigscience

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed Megatron-DeepSpeed-tr1-13B
cd Megatron-DeepSpeed-tr1-13B
git checkout tr1-13B
pip install -r requirements.txt
pip install -e .
mkdir data
cd data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

`apex` and `deepspeed` build require an instance w/ beefy cpu and internet (unless cloned beforehand), so continue on the `prepost` partition:

apex 和 deepspeed 的构建需要一台配置强大的 CPU 和互联网连接的实例（除非事先克隆），因此请继续在 prepost 分区上进行操作：

```
ssh jean-zay-pp
conda activate tr1-13B
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/microsoft/DeepSpeed DeepSpeed-big-science
cd DeepSpeed-big-science
git checkout big-science
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log

cd $six_ALL_CCFRWORK/code/tr1-13B
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log

#cp $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base/start-tr1-13B ...

```


## Architecture
## 架构

Config:

```
NLAYERS=40
NHIDDEN=5120
NHEADS=32
FFN_HIDDEN_SIZE=20480

#    --ffn_hidden_size $FFN_HIDDEN_SIZE \
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    [...]
    "
```

Sanity check:
```
$ VOCAB_SIZE=50257 NLAYERS=40 NHIDDEN=5120 NHEADS=32 SEQ_LEN=2048; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"
Model size: 13B
```



## Sequence Length

Default Megatron-LM language model with 2048 tokens sequence length

默认的 Megatron-LM 语言模型使用 2048 个 token 的序列长度。

```
SEQ_LEN=2048

    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \

```


## Global batch size（全局 batch size 大小）

GBS = Global Batch Size

Use a schedule:

使用以下任务计划

- start from 32k tokens (gbs=16)

  从 32k tokens 开始（全局 batch size 大小为 16）
- increase linearly to 2048k (gbs=1024) over 5M samples (for a total of ~10B tokens / 5k steps)

  在 500 万个样本中线性增加到 2048k（gbs=1024）（总共大约 10B tokens/5k 个步骤）
- then continue at 2048k  (gbs=1024) for 145M samples (290B tokens / 145K steps)

  然后在 1.45 亿个样本中继续保持 2048k（gbs=1024）（总共 2900 亿个 tokens/145k 个步骤）

Total: 300B tokens (150K steps)

总计：300B tokens（150K steps）

Note: the training script wasn't updated when we flipped seqlen/gbs from 1024/2048 to 2048/1024, so we are currently planning to train for 300K steps (samples) and 600B tokens. But since longer doesn't impact anything, we will just stop at half the time. I updated the document to use the right 150K number so we don't repeat this mistake in the next training.

注意：当我们将 seqlen/gbs 从 1024/2048 翻转为 2048/1024 时，训练脚本没有进行更新，因此我们目前计划进行 300K 个步骤（样本）和 600B 个 tokens 的训练。但由于更长时间不会对任何事情产生影响，我们将在半程停止。我已经更新了文档，使用正确的 150K 数字，以避免在下一次训练中重复这个错误。

syntax(语法):
```
--rampup-batch-size <start batch size>  <batch size increment> <ramp-up samples>
```

At seqlen 2048 (1k tokens is bs=1), we get:

在序列长度为 2048（当 batch size 为 1 时，1k 个标记）的情况下，我们得到：

```
    --rampup-batch-size 16 16 5_000_000 \
    --global-batch-size 1024 \
```

This means it will start with global batch size 16 and over 63 (`(1024-16)/16`) intervals will increase the batch size by 16 linearly to 1024.

这意味着它将从全局的 batch size 大小 16 开始，并且在 63 个间隔内（(1024-16)/16）以线性方式将批大小逐渐增加到 1024。

79365 (`5_000_000/63`) is the number of samples before the next GBS increment. That is we run at GBS=16 for 79365 samples, or 4960 steps (`79365/16`). Then we run at GBS=32 for 79365 samples, or 2480 steps. Then 1653 steps at GBS=48, 1240 at GBS=64, etc....

79365（5_000_000/63）是下一个 GBS 增量之前的样本数量。也就是说，在 79365 个样本或 4960 个步骤（79365/16）中，我们使用 GBS=16 运行。然后在 79365 个样本或 2480 个步骤中使用 GBS=32 运行。然后使用 GBS=48 运行 1653 个步骤，使用 GBS=64 运行 1240 个步骤，依此类推...

Notes(注意):
* `--rampup-batch-size` requires the use of `--train-samples` and can't be used with `--train-iters`.

  --rampup-batch-size 需要使用 --train-samples 并且不能与 --train-iters 一起使用。
* global batch size has to be divisible by micro-batch-size * DP_SIZE

  全局 batch size 大小必须能够被微 batch size 大小（micro-batch-size）乘以 DP_SIZE 整除。

Important:  the software will fail if GBS is not divisible by `MBS * DP_SIZE`.

重要提示：如果 GBS 不能被 MBS * DP_SIZ E整除，软件将会失败。

Though Jared's recommendation is to use MBS=1 and then it's much easier to match GBS/DP_SIZE even at GBS=16.

尽管 Jared 的建议是使用 MBS=1，这样即使在 GBS=16 时，也很容易匹配 GBS/DP_SIZE。

`DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE)`

Since the increments are in GBS=16, we can do only DP_SIZE=16, which means that at most we can use 32 nodes (`32*4/(4*2)=16`).

由于增量是以 GBS=16 为单位的，我们只能使用 DP_SIZE=16，这意味着最多可以使用 32 个节点（32*4/(4*2)=16）。

Once GBS reaches 1024, we can use up to 8192 GPUs (1024*2*4), so we will be able to switch to 64 nodes or may be even 128 nodes (4 gpus each). We can't use any number of nodes between 64 and 128 though, because the number has to be 2**X. So 96 nodes won't work, because it has a multiplier of 3 there.

一旦 GBS 达到 1024，我们最多可以使用 8192个GPU（102424），因此我们可以切换到64个节点，甚至可能是128个节点（每个节点4个GPU）。但是，在64和128之间我们不能使用任何节点数，因为节点数必须是2的幂次方。因此，96个节点将无法工作，因为其中有一个乘数为3。



## Checkpoints
## 检查点

We need the checkpoints(我们需要检查点的原因如下):

1. in order to be able to resume the training when the training is prematurely stopped for whatever reason.

   为了能够在训练过程中出现任何意外停止的情况下恢复训练。
3. In addition a special saving schedule has been requested by the interpretabity group.

   此外，可解释性团队（Interpretabity Group）还提出了特殊的保存计划要求。
Because there are 3 different schedules, and Megatron-LM has only fixed checkpoint saving schedule, we will need 3 different run scripts, to be launched in a sequence, each starting once the previous has finished.

由于有 3 个不同的任务执行计划，而 Megatron-LM 只有固定的检查点保存任务执行计划，我们将需要 3 个不同的运行脚本，按顺序启动，每个脚本在前一个脚本完成后开始运行。

1. steps 1-100 - 10 checkpoints, interval 10 steps

   步骤 1-100 - 10 个检查点，间隔 10 步
3. steps 101-1000 - 50 checkpoints, interval 18 steps

   步骤 101-1000 - 50 个检查点，间隔 18 步
5. steps 1001-150K - 100+ checkpoints, interval 1500 steps

   步骤 1001-150K - 100+ 个检查点，间隔 1500 步
7. if still needed, can continue with schedule 3

   如有需要，可以继续使用计划 3
note: the interoperability study doesn't care for checkpoints in the range of 1k-20k, so we only save those to be able to restart the training.

注意：可解释性研究并不关心 1,000 至 20,000 范围内的检查点，所以我们只保存这些检查点以便能够重新启动训练。

It'd have been
```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then TRAIN_ITER=100    SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then TRAIN_ITER=1000   SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then TRAIN_ITER=150000 SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi
    --train-iter $TRAIN_ITER  \
    --save-interval $SAVE_INTERVAL  \
```

Unfortunately, `--rampup-batch-size` can't work with `--train-iter` and we have to use `--train-samples` instead. It has to be fixed through all of trainings and can't be changed, otherwise resume from checkpoint will break.

很不幸，--rampup-batch-size 无法与 --train-iter一 起使用，我们必须使用 --train-samples 来代替。这个问题需要在所有的训练过程中修复，并且不能更改，否则从检查点恢复将会出错。

So the only thing left is to use `--exit-interval` which is in steps.

所以唯一剩下的选择是使用步数间隔 --exit-interval。

Which gives us the three rounds:

这样我们可以将训练分为三个阶段：

```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then EXIT_INTERVAL=100 SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then EXIT_INTERVAL=900 SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then                   SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

    --train-samples 150_000_000 \
    --exit-interval $EXIT_INTERVAL \
    --save-interval $SAVE_INTERVAL  \
```

`--exit-interval` counts steps only for the current run, regardless of previous steps. So to stop at effective step 1000, the second round we tell it to exit at 900 (the first round did the first 100).

--exit-interval 只计算当前运行的步数，不考虑之前的步数。因此，为了在有效步骤 1000 处停止，第二轮训练我们需要告诉它在 900 步时退出（第一轮已经完成了前 100 步）。

And unfortunately, this proved to be not supported by Megatron-LM either at the moment. There are a few possible ways to approach this:

很遗憾，在目前的 Megatron-LM 中也不支持这种方式。有几种可能的解决方法可以尝试：

1. One approach is to simply use 3 independent trainings, while using the same `--seed ` and just have `--exit_interval` as above. Though after each training moving the checkpoints away.

   一种方法是简单地进行3个独立的训练，同时使用相同的 --seed 并设置与上述相同的 --exit-interval。但在每次训练之后，需要将检查点移出。

3.
XXX: Also megatron code could be extended to implement `--exit-samples` - so sample-based exit strategy

XXX: Megatron的代码也可以扩展来实现 --exit-samples，即基于样本数量的退出策略。

4. Yet another approach is to do it manually. Kill the training after 100, and then restart and kill after 900 iterations, while changing the save interval, and manually fixing up the `checkpoints/latest` to point to the correct checkpoint - since the manual killing might have a few extra checkpoints. So the recipe to follow:

另一种方法是手动操作。在进行 100 次迭代后停止训练，然后重新启动并在 900 次迭代后停止，同时更改保存间隔，并手动修正 checkpoints/latest 以指向正确的检查点 - 因为手动停止可能会有几个额外的检查点。所以遵循以下步骤：

```
ROUND=1
if   [[ ${ROUND} == 1 ]]; then SAVE_INTERVAL=10
elif [[ ${ROUND} == 2 ]]; then SAVE_INTERVAL=18
elif [[ ${ROUND} == 3 ]]; then SAVE_INTERVAL=1500
else echo "invalid ROUND: $ROUND"
fi

    --train-samples 150_000_000 \
    --save-interval $SAVE_INTERVAL  \
```

(could also do it with 3 parallel jobs by using the same seed!)

(也可以通过使用相同的种子在3个并行作业中执行此操作！)


```
--seed 42
```

Therefore do this manually:

因此，请手动完成以下操作：

0.
* delete the old checkpoints `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints`

  删除旧的检查点 $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints。

1.

* set to `ROUND=1`
* `sbatch tr1-13B-round1.slurm`
* run for 100+ steps
* scancel the job
* clean up `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints` to remove any checkpoints beyond 100
* make sure `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/latest` contains 100


2.

* set to `ROUND=2`
* `sbatch tr1-13B-round1.slurm`
* run for the additional 900+ steps (it's incremental, so the script already knows it started at 100)
* scancel the job
* clean up `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints` to remove any checkpoints beyond 1000
* make sure `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/latest` contains 1000


3.

* set to `ROUND=3`
* `sbatch tr1-13B-round1.slurm`
* run normally


Because it'd be potentially too demanding to export TBs of data and the intended users might not be even able to download all that data, most likely we will need to run the interpretabity post-analysis experiments on JZ and send the reports to those who need the reports.

由于导出数 TB 的数据可能过于耗费资源，而且预期的用户可能无法下载所有这些数据，因此我们很可能需要在 JZ 上运行可解释性后续分析实验，并将报告发送给需要报告的人员。

Megatron-LM resumes from the most recent checkpoint by default. Does it need the exact path or does it auto-discover the latest checkpoint by default.

Megatron-LM 默认情况下会从最新的检查点恢复训练。它是否需要精确的路径或自动发现最新的检查点取决于具体设置。

```
--load path_to_check_point \
```

Remi suggests 100TB on SCRATCH shouldn't be a problem.

Remi 建议在 SCRATCH 上使用 100TB 不应该成为问题。


## Optimizer
## 优化器

- AdamW,  β1=0.9, β2=0.999 eps=1e−8
- learning rate:
   * peak=1e-4
   * warmup over 2000 steps
   * cosine decay for learning rate down to 10% of its value, over 260B tokens (after 260 billion tokens, training continues at 10% of the original learning rate)

     余弦衰减学习率：在训练过程中，学习率按余弦函数进行衰减，直到其值的 10%。这个衰减过程在训练达到 2600 亿个 tokens 后开始生效，之后以较低的学习率继续训练。
- clipping by global norm of 1 (as in GPT-3)

  对梯度进行全局范数裁剪，裁剪阈值为1（与 GPT-3 类似）。
- weight decay of 0.1

  权重衰减（weight decay）为0.1

We need lr-decay in samples, so tokens2samples = 260B / 2048 = 126_953_125

我们需要以样本为单位进行学习率衰减，因此 tokens2samples = 2600亿 / 2048 = 126_953_125。

We need lr-warmup in samples, so doing the math again as in checkpoints

我们需要以样本为单位进行学习率热身（lr-warmup），因此可以再次进行计算，如在检查点中所示：

2000=160*12+80

so we will get to 2000 in 216_320 samples `16*160*12*(12+1)/2+16*13*80`

因此，为了在 216,320 个样本内达到 2000 步，我们可以使用以下计算：16*160*12*(12+1)/2+16*13*80。

```
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
```


## Logging(日志记录)

For now enable all tensorboard features, later we might decide to not log it all.

目前启用所有 TensorBoard 功能，稍后我们可能决定不记录所有内容。

We are logging:

我们正在记录以下内容：

- lr (enabled by default)
- bs (enabled)
- loss (always)
- loss-scale (log_loss) (enabled by default)
- grad-norm (always)
- num-zeros (always)
- param-norm (always)
- timers (enabled)
- validation loss (always)
- validation ppl (perplexity) (enabled)

almost all of these are also logged as a comparison to consumed_train_samples

几乎所有这些内容都是与 consumed_train_samples 进行比较后记录的。

XXX: nice to have:

XXX：可选项：

- throughput - Tflops/gpu or tokens

  吞吐量（throughput）：以每个 GPU 的 Tflops 或 tokens 为单位进行记录。


**Tensorboard config**:

```
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard

    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
```

**CodeCarbon config**:

```
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon

    --codecarbon-dir $CODECARBON_PATH \
```



**Training logs**

All training logs are piped into `$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs/main_log.txt`.

所有的训练日志都被导入到 $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs/main_log.txt 文件中。



## Exporting

Before starting training create cloned git repos to where output data will go.

在开始训练之前，创建克隆的 Git 仓库，用于存放输出数据。

The last 4 should all be git repo clones

最后的 4 个都应该是 Git 仓库的克隆。

```
DATA_OUTPUT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
CODECARBON_PATH=$DATA_OUTPUT_PATH/codecarbon
LOGS_PATH=$DATA_OUTPUT_PATH/logs
```

I created 4 repos at https://huggingface.co/bigscience/ and now we can clone those as the dirs data will be output into:

我在 https://huggingface.co/bigscience/ 创建了 4 个仓库，现在我们可以克隆这些仓库作为输出数据的目录：

```
cd $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B
git clone https://huggingface.co/bigscience/tr1-13B-checkpoints checkpoints
git clone https://huggingface.co/bigscience/tr1-13B-tensorboard tensorboard
git clone https://huggingface.co/bigscience/tr1-13B-codecarbon codecarbon
git clone https://huggingface.co/bigscience/tr1-13B-logs logs
```

If this is your first time running git-lfs on this system, you need to init it once:

如果这是您第一次在该系统上运行 git-lfs，您需要初始化它一次：

```
module load git-lfs
git lfs install
```

Most of the data types we are going to sync will be large or huge, and most are already lfs-tracked by default, so no setup is required. Except our log file which too can grow large, so we need to set it up:

我们将要同步的大部分数据类型都是大型或巨大的，并且大多数已默认使用 LFS 进行跟踪，因此不需要进行设置。但是我们的日志文件也可能变得很大，因此我们需要对其进行设置：


```
cd logs
git-lfs track *.txt
git commit -m "large text files" .gitattributes
git push
```

### Cronjobs to auto-sync the hub

Now we just need a cronjob to automatically do for each type of data to export:

现在我们只需要为每种类型的数据自动创建一个 cron 作业来导出：

```
cd checkpoints
git add */*.pt
git commit -am "new data"
git push
```

This job is performed automatically by `hub-sync.py`. For full details see: [Automated upload to the hub](../../data/export.md#automated-upload-to-the-hub).

这个任务由 hub-sync.py 自动执行。有关详细信息，请参阅：Automated upload to the hub。


**Weights checkpoints**

Currently, we aren't exporting checkpoints.

目前，我们不导出检查点。

**Tensorboard**

Here is the slurm script to sync the tensorboard data: [tr1-13B-hub-sync-tensorboard.slurm](./tr1-13B-hub-sync-tensorboard.slurm)

这是一个用于同步 TensorBoard 数据的 Slurm 脚本：[tr1-13B-hub-sync-tensorboard.slurm](./tr1-13B-hub-sync-tensorboard.slurm)

**CodeCarbon**

Currently the feature is not enabled, so there is nothing to log.

当前该特性未启用，因此没有需要记录的内容。

**Log of logs(日志记录)**

Let's also create a log of logs. We will pipe all the logs in there and also the various status reports - e.g. while SLURM is queued the training and it's not running.

我们还将创建一个“日志记录”的日志文件。我们将把所有的日志信息和各种状态报告都导入其中，例如当 SLURM 处于排队状态时，训练尚未运行。

Here is the slurm script to sync the raw logs data: [tr1-13B-hub-sync-logs.slurm](./tr1-13B-hub-sync-logs.slurm)

这是用于同步原始日志数据的 SLURM 脚本： [tr1-13B-hub-sync-logs.slurm](./tr1-13B-hub-sync-logs.slurm)

The main source of logs is the training scripts. The logs are gathered via

主要的日志来源是训练脚本。通过以下方式收集日志：
```
$CMD ... 2>&1 | tee -a $LOGS_PATH/main_log.txt
```
in the training slurm script.

在训练的 SLURM 脚本中。

XXX: we could also add various other diagnostics appended to the main log file. e.g. shared memory, etc.

XXX: 我们还可以将其他各种诊断信息追加到主日志文件中，例如共享内存等。

## Deepspeed config(Deepspeed 配置)

Using Deepspeed's activation checkpointing to use a lot less GPU memory

使用 DeepSpeed 的激活检查点技术可以大大减少 GPU 内存的使用量。

```
    --deepspeed-activation-checkpointing \
```

Possible extras:

可能的附加信息：

- Enabling `"contiguous_memory_optimization": true,` can help to reduce memory fragmentation, but it requires￼setting `number_checkpoints`. This should be set to be equal to number of transformer blocks per pipeline stage times the number of pipeline parallel stage. Samyam says: Full disclaimer: I have only used this with ZeRO but not with pipeline parallelism. But by setting the number_checkpoints as described, it should work for PP too. The benefit of using it is usually only apparent when running very close to the memory limit.

启用 "contiguous_memory_optimization": true 可以帮助减少内存碎片化，但需要设置 number_checkpoints 参数。该参数应设置为每个 pipeline 阶段的 Transformer 块数乘以管道并行阶段的数量。Samyam 表示：完全免责声明：我只在使用ZeRO 时使用过此功能，而没有使用管道并行性。但是，通过按照所述设置 number_checkpoints 参数，它应该也适用于管道并行。使用此功能的好处通常只在接近内存限制时才显现出来。


## Dataset(数据集)

- Full 304.2M version (529GB) : `$six_ALL_CCFRWORK/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k`

We are using English-only subset of [the OSCAR dataset](https://huggingface.co/datasets/oscar) with full documents (*not* individual sentences).

我们正在使用 OSCAR 数据集的仅限英语的子集，其中包含完整的文档（而不是单个句子）。

We have about 300M records in 1.2TB of jsonl data (about 3/4 of which are smaller than 1K tokens), which amounts to about 280B tokens (estimated at about 4.5chars/word).

我们拥有大约 3 亿条记录，占据了 1.2TB 的 JSONL 数据（其中大约 3/4 的记录长度小于 1K 个标记），总计大约 2800 亿个标记（估计每个词约有 4.5 个字符）。

Megatron's preprocessing tool indexes everything and then at training time the Dataloader serves chunks of the desired fixed sequence length (2048 tokens in our case).

Megatron 的预处理工具对所有内容进行索引，然后在训练时，Dataloader 会提供所需的固定序列长度的数据块（在我们的情况下为 2048 个标记）。

For more information on the pre-processing process and various estimations see: [OSCAR](../../data/oscar/README.md).

有关预处理过程和各种估算的更多信息，请参阅：OSCAR。


## Dealing with 20h SLURM limit(处理 SLURM 20h 限制(最多运行 20h 的限制))

First, let's ensure we save a checkpoint just before SLURM kills the job

首先，让我们确保在 SLURM 终止作业之前保存一个检查点。

Let's try 19:50 1190=60*20-10

我们来计算 19:50，这个时间是通过减去 10 分钟（10 minutes）从 20 小时（20 hours）中得到的。

```
    --exit-duration-in-mins 1190 \
```

For the bigger models 10min might not be long enoug to finish an iteration (assume the limit hits right as one starts) and write out a checkpoint.

对于更大的模型来说，10 分钟可能不足以完成一次迭代（假设限制正好在一次迭代开始时触发），并将检查点写入磁盘。

Then we need to figure out how to schedule the next slurm job as soon as the currently running one is over in 20h.

那么我们需要找出如何在当前运行的 SLURM 作业结束后的 20 小时内安排下一个 SLURM 作业。

We will use job arrays, to solve this. Let's start with just 10 such jobs:

我们将使用作业数组（job arrays）来解决这个问题。让我们从仅有 10 个这样的作业开始：

```
sbatch --array=1-10%1 tr1-13B-round1.slurm
```

`%1` limits the number of simultaneously running tasks from this job array to 1, since we want them to run in a sequence.

`%1` 限制了该作业数组中同时运行的任务数量为 1，因为我们希望它们按顺序运行。

Alternatively, as always this param can be part of the script:

另外，与往常一样，此参数也可以作为脚本的一部分：

```
#SBATCH --array=1-10%1
```

## Crontab(定时任务)

JZ doesn't have a user-accessible crontab facility, so we have to emulate it with a self-restarting slurm job that polls some dir for new jobs to run. For full details on how this works please see [Crontab Jobs](../../jz/crontab/).

JZ 没有用户可访问的 crontab 功能，因此我们必须使用自重启的 Slurm 作业来模拟它，该作业会轮询某个目录以查找要运行的新作业。有关此工作原理的完整详细信息，请参阅 [Crontab Jobs](../../jz/crontab/).。

But to use it simply put your slurm scripts into either:

要使用它，只需将您的 Slurm 脚本放入以下目录之一：

```
$six_ALL_CCFRWORK/cron/cron.hourly
$six_ALL_CCFRWORK/cron/cron.daily
```

and the jobs will be run on hourly or daily basis. This is similar to Linux's `/etc/cron.*` setup. Except the jobs aren't guaranteed to start on the hour, but should be around that time.

并且作业将按小时或每天的频率运行。这类似于 Linux 的 /etc/cron.* 设置。不过，这些作业不能保证在整点开始，但应该在大致相同的时间附近启动。

Currently we have:

目前我们有：

```
ls -1 $six_ALL_CCFRWORK/cron/cron.hourly/*slurm
tr1-13B-hub-sync-logs.slurm
tr1-13B-hub-sync-tensorboard.slurm
tr1-13B-slurm-status.slurm
```

The first 2 sync log files to the hub and the last one monitors the health of the training and alerts of any problems.

前两个同步日志文件到中央服务器，而最后一个监控训练的健康状况，并在出现任何问题时发出警报。


## Estimated run time(预计运行时间)

Best case scenario when training 24/7 on 64 nodes with 4 gpus each:

在每个节点上使用 4 个 GPU，全天候（24/7）使用 64 个节点进行训练的最佳情况：

```
$ python -c 'Btokens=300; Bmodel=13; n_gpus=256; Tflops=45; \
print(f"{Btokens*1e9*8*Bmodel*1e9/(n_gpus*Tflops*1e12*60*60*24):0.2f} days")'
31.35 days
```

You will find the detailed explanation of the estimation formula [here](../../math/README.md#estimate-model-training-time).

您可以在这里找到有关估计模型训练时间的公式的详细说明 [here](../../math/README.md#estimate-model-training-time).

The training was much slower in the first 10k steps because of the batch size rampup, where the pipeline was very inefficient.

由于批量大小逐渐增加，最初的 10,000 个步骤的训练速度较慢，其中管道效率较低。

And then we were only able to use 20h slurm jobs, with unpredictable gaps of wait time in between (1-30 hours!), so it's impossible to predict when the finish line will be finished.

然后，我们只能使用每个作业 20 小时的 Slurm 作业，并且在它们之间存在不可预测的等待时间间隔（1-30 小时！），因此无法预测何时能够完成训练。

## Memory usage(内存使用情况)

During training currently we use 256GB (8x 32GB gpus) per each full replica (TP=2 + PP=4), the rest are ZeRO-DP. So if we throw x times more GPUs we just speed things up by having more 2-node replicas.
The required memory breakdown:

在当前的训练过程中，每个完整的副本（TP=2 + PP=4）使用了 256GB 的内存（8 个 32GB 的 GPU），其余部分使用了 ZeRO-DP 技术。

1. 4B for fp32 weights
2. 2B for fp16 weights
3. 8B for optimizer states.
4. 4B for gradients (we don't save these in the checkpoint)
5. plus memory for activations and temps, which total majorly depends on the seqlen and mini batch size - and since we use activation checkpointing this memory need is quite small.

   另外还有用于激活值和临时变量的内存，这个内存需求主要取决于序列长度和小批量大小 - 由于我们使用了激活值检查点技术，因此这个内存需求相对较小。

Total: 234GB (18*13) plus activations and temps memory. So we are close to 256GB here.

总计：234GB（18 * 13），加上激活值和临时变量的内存。所以我们接近 256GB。

Activation memory would have been much much bigger if it weren't for activation checkpointing.

如果没有使用激活值检查点技术，激活值的内存需求将会大得多。


## Checkpoint Back Up(检查点备份)

To copy multiple checkpoints excluding optimizer states. First move the desired checkpoints to back up to some dedicated dir, e.g. `tr1-13B-round2/checkpoints`, then copy just the needed files:

要复制多个检查点，但排除优化器状态，请先将要备份的检查点移动到专用目录（例如 tr1-13B-round2/checkpoints），然后只复制所需的文件：

```
srun -p prepost  -A six@cpu --time=20:00:00 --pty bash
mkdir to-upload
rsync -acvhu --no-compress --info=progress2 --exclude "zero*pt" tr1-13B-round2/checkpoints/ to-upload
```

then to back those up:

然后将它们进行备份：

```
cp -arun $six_ALL_CCFRSCRATCH/checkpoints/to-upload/* $six_ALL_CCFRSTORE/checkpoints/tr1-13B
```


**Final checkpoint with optimizer states(包含优化器状态的最终检查点):**

```
mkdir $six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim
cp -arun $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/checkpoints/global_step168000 $six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim/
```

This is the final checkpoint, that can be resumed from at will:

这是最终的检查点，可以随时恢复训练：

```
$six_ALL_CCFRSTORE/checkpoints/tr1-13B-with-optim/global_step168000
```

Here is the corresponding log:

这是相应的日志记录：

```
 iteration   168000/  311541 | consumed samples:    153013584 | elapsed time per iteration (ms): 13248.2 | learning rate: 1.000E-05 | global batch size:  1024 | lm loss: 2.376641E+00 | loss scale: 131072.0 | grad norm: 19767.052 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms)
--------------------------------------------------------------------------------------------------
 validation loss at iteration 168000 | lm loss value: 2.342049E+00 | lm loss PPL: 1.040253E+01 |
--------------------------------------------------------------------------------------------------
```

## Checkpoint Conversion and Upload(检查点转换和上传)

**Important**: there was a bug in the converter on the transformers side, so we need this fix:

**重要提示**：converter 在 transformers 库中存在一个错误，我们需要修复此问题：

https://github.com/huggingface/transformers/pull/13735
if it's not merged yet, install this branch first. If it's already merged just make sure you use `transformers@master` - XXX: I will update the script to require a specific version once a new version of transformers is released.

如果该分支尚未合并，请首先安装此分支。如果已经合并，请确保使用 transformers@master - XXX：一旦发布了新版本的 transformers，我将更新脚本，要求使用特定版本。

Open a long running interactive shell:

打开一个长时间运行的交互式 Shell：
```
srun -p compil --cpus-per-task=40 -A six@cpu --time=6:00:00 --pty bash
```
then convert:

然后进行转换：
```
cd $six_ALL_CCFRSCRATCH/checkpoints/to-upload
time find * -maxdepth 0 -type d -name "global_step*" -exec $six_ALL_CCFRWORK/code/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder {} --output_folder hf-fixed/{} \;
```

It takes about 100sec per 26GB checkpoint.

每个 26GB 的检查点需要大约 100 秒。

The results will be all under `hf/`.

所有结果将会存储在 hf/ 目录下。

Now to uploading to the hub.

现在开始上传到 Hub

Prepare the target dir:

准备目标目录：

```
#git -c http.extraHeader="Authorization: Basic " clone https://huggingface.co/bigscience/tr1-13B-checkpoints/

cd tr1-13B-checkpoints


huggingface-cli lfs-enable-largefiles .

git config --unset user.email
~/prod/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
```
We are going to put each checkpoint into its own branch with the same name.

我们将每个检查点放置在以相同名称命名的单独分支中。

```
mv ../hf/global_step* .
time find * -maxdepth 0 -type d -name "global_step*" -exec git checkout main \; -exec git checkout -b {} \; -exec git add {} \; -exec git commit -m "add {}" \; -exec git push --set-upstream origin {} \;
git checkout main
```

Fixing up failed pushes / verifying that all pushes went through, re-pushing if needed

修复推送失败 / 验证所有推送是否成功，如有需要则重新推送。

```
git branch | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; git push --set-upstream origin $1]'
```

If `git push` fails re-run with: `GIT_TRACE=1 GIT_TRANSFER_TRACE=1 GIT_CURL_VERBOSE=1 git push` to see what the actual error is.

如果 git push 失败，请使用以下命令重新运行：GIT_TRACE=1 GIT_TRANSFER_TRACE=1 GIT_CURL_VERBOSE=1 git push，以查看实际的错误信息。


OK, the branch-per-checkpoint hub repo proved to be very difficult to upload and even more so using it after the upload.

好的，针对每个检查点的分支式 Hub 存储库在上传和上传后的使用方面证明非常困难。

So let's try GCS bucket:

所以，让我们来尝试使用 GCS 存储桶（Google Cloud Storage）：

```
gcloud auth login
gcloud config set project bigscience
gsutil cp -r hf-fixed/* gs://bigscience-backups/tr1-13B/checkpoints/

```
or via rsync:
```
gsutil -m rsync -r hf-fixed/* gs://bigscience-backups/tr1-13B/checkpoints/
```

```
start-prod
cd /gpfsssd/scratch/rech/six/commun/checkpoints/to-upload/
gsutil -m rsync -r hf-fixed1/* gs://bigscience-backups/tr1-13B/checkpoints/

```

or if needed to speed up the upload via multiple parallel copies open 2 `srun` instances and in one:

如果需要通过多个并行拷贝来加快上传速度，请打开两个 `srun` 实例，并在其中一个中执行以下操作：

```
gsutil cp -r hf-fixed1/* gs://bigscience-backups/tr1-13B/checkpoints/
```
and in another:

在另一个实例中：

```
gsutil cp -r hf-fixed2/* gs://bigscience-backups/tr1-13B/checkpoints/
```

can't use `rsync` with multiple sources - can only rsync a single dir.

无法使用 `rsync` 同时处理多个源文件/目录 - 只能同步单个目录。

Later fixing `config.json` to include the correct `gelu_fast` activation correction and rsyncing the GCS bucket.

后续需要修复 config.json 文件，包括正确的 gelu_fast 激活修正，并进行 GCS 存储桶的同步。

(moved all the hf-fixed sub-dirs into a new folder `checkpoints`)

(将所有 hf-fixed 子目录移动到一个名为 checkpoints 的新文件夹中。)

```
start-prod
cd /gpfsssd/scratch/rech/six/commun/checkpoints/to-upload/
perl -pi -e 's|gelu|gelu_fast|' checkpoints/*/config.json
gsutil -m rsync -x ".*bin$" -r checkpoints gs://bigscience-backups/tr1-13B/checkpoints
```
this is really fast since we exclude the checkpoint files (`-x ".*bin$"`)

这样做非常快，因为我们排除了检查点文件（-x ".*bin$"）。


## Other backups(其他备份)

Logs:

```
mkdir $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/
tar -zcvf $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/tensorboard.tgz $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/tensorboard
tar -zcvf $six_ALL_CCFRSTORE/checkpoints/tr1-13B-logs/logs.tgz $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs
```

note: codecarbon wasn't ready during this training, so nothing to back up there.

注意：在这次训练中，CodeCarbon还没有准备就绪，所以没有相关的备份。


## Exports

- GCS https://console.cloud.google.com/storage/browser/bigscience
- The Hub https://huggingface.co/bigscience


## Training scripts(训练脚本)

The training script is:

训练脚本是：

- [tr1-13B-round1.slurm](./tr1-13B-round1.slurm)

We also have:

我们还有以下：

- [tr1-13B-short.slurm](./tr1-13B-short.slurm)

which is a very small model to do quick testing and debug, but otherwise the same as the main script.

这是一个非常小的模型，用于进行快速测试和调试，但除此之外与主要脚本相同。

The scripts are located at:

这些脚本位于以下位置：

```
cd $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base
```

When no jobs are scheduled, currently we launch the main training script using:

当没有作业被调度时，目前我们使用以下方式启动主要的训练脚本：

```
sbatch --array=1-5%1 tr1-13B-round1.slurm
```
This will schedule 5 20h-trainings which will run one at a time, once the scheduler yields to the request, with unknown wait time in between each job.
If there is a job running already, **do not use the above command** as we can't have 2 trainings overlap. If there is a training already running you can:

这将安排 5 个 20 小时的训练任务，每次只运行一个任务，一旦调度程序接受请求，之间的每个任务之间会有未知的等待时间。
如果已经有一个作业正在运行，请不要使用上述命令，因为我们不能让两个训练重叠。如果已经有一个训练正在运行，您可以：

1. either tell `sbatch` to start the new job once the currently running job succeeds, using:
1. 或者告诉 `sbatch` 在当前运行的作业成功后启动新作业，使用以下命令：

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID --array=1-5%1 tr1-13B-round1.slurm
```

Where `CURRENTLY_RUNNING_JOB_ID` is the job being reported running. For example if the report of the last job is:

其中 CURRENTLY_RUNNING_JOB_ID 是指正在报告运行的作业。例如，如果上一个作业的报告如下：
```
[2021-08-16 22:08:01] tr1-13B-round3 is running for 18:15:59 since 2021-08-16T03:52:02 (711114_4 on 'gpu_p13' partition (r7i4n[1-7],r7i7n[1-8],r8i0n0,r8i5n[3-8],r8i6n[0-8],r9i0n8,r9i1n[0-8],r9i2n[7-8],r9i3n[0-8],r9i4n[0-8],r9i5n[0-2])
```
then the currently running job ID is `711114_4`. You can also gather the same info about the current scheduler status using `squeue`:

那么当前正在运行的作业 ID 是 711114_4。您也可以使用 squeue 收集有关当前调度器状态的相同信息。

```
squeue --user=$(getent group six | cut -d: -f4) | grep tr1-13B
```

2. you could also see how much time is left before the current job finished (based on training log files) and then pass that many hours to `sbatch`. For example, if the job has **less** than 2 hours to run, but more than 1 hour, you want to launch it `now+2hours` from now:

您还可以查看当前作业完成所需的剩余时间（基于训练日志文件），然后将相应的小时数传递给 sbatch。例如，如果作业剩余时间不足 2 小时，但超过 1 小时，您希望从当前时间开始的 2 小时后启动它：


```
sbatch --begin now+2hours --array=1-5%1 tr1-13B-round1.slurm
```

Using `--dependency` may lead to shorter wait times, since if the time passed to `--begin` allows even for a few minutes of delay since the stopping of the last job, the scheduler may already start some other jobs even if their priority is lower than our job. That's because the scheduler ignores any jobs with `--begin` until the specified time arrives.

使用 --dependency 参数可能会导致更短的等待时间，因为如果传递给 --begin 的时间允许在上一个作业停止后有几分钟的延迟，调度器可能会开始执行其他一些作业，即使它们的优先级低于我们的作业。这是因为调度器会忽略任何具有 --begin 参数的作业，直到指定的时间到达。

## On Call

When a person is on call, they need to watch that the training is either running or scheduled to run. If neither is happening they need to schedule a new training. When this situation occurs the log file will report:

当一个人值班时，他们需要确保训练要么正在运行，要么已安排运行。如果两者都没有发生，他们需要安排一个新的训练。当出现这种情况时，日志文件将会报告：

```
***ALERT: tr1-13B-round3.slurm is not RUNNING or SCHEDULED! Alert someone at Eng WG***

***警报：tr1-13B-round3.slurm 不在运行或计划中！请通知工程工作组的相关人员***
```

An email alert is sent as well to `bigscience-jean-zay@groups.google.com`.

同时也会发送电子邮件警报至 bigscience-jean-zay@groups.google.com。

The next section explains how to watch the logs.

下一部分将解释如何监视日志。

Other than waiting for the watchdog which runs once an hour, one can immediately see if anything is scheduled with:

除了等待每小时运行一次的看门狗（定时运行的程序或脚本）之外，您可以立即查看是否有任何计划安排，方法是：

```
$six_ALL_CCFRWORK/code/tr1-13B/bigscience/tools/slurm-status.py --job-name tr1-13B-round3
```

If for some reason the training is not scheduled or running, to schedule a new training:

如果由于某种原因训练没有被调度或正在运行，需要安排一个新的训练：

```
cd $six_ALL_CCFRWORK/code/tr1-13B/bigscience/train/tr1-13B-base
sbatch --array=1-5%1 tr1-13B-round1.slurm
```

This will schedule a job array of 5 jobs of 20h each, so if all goes well, that's at least 4 days of not needing to do anything other than being on the lookout for potential crashes.

这将安排一个由 5 个作业组成的作业数组，每个作业持续 20 小时，因此如果一切顺利，至少可以连续 4 天不需要做任何事情，只需要留意潜在的崩溃情况。

XXX: need a troubleshooting section, but elsewhere in the document that is not this training specific.

XXX：需要一个疑难解答部分，但是在文档的其他地方，而不是这个特定的训练部分。

1. if one of the nodes gets a corrupted gpu, and the training crashes there is a risk that the next job in the training will get allocated the same node, in which case it'll crash again. We need a method to identify which node is corrupted, report that to assist@idris.fr so they know to fix it and exclude this node from the slurm job by adding a list of nodes to exclude as following:

1. 如果其中一个节点的 GPU 发生损坏，并且训练在该节点上崩溃，那么下一个作业可能会被分配到相同的节点，导致再次崩溃。我们需要一种方法来确定哪个节点发生了损坏，并将其报告给 assist@idris.fr，以便他们知道需要修复它，并通过添加要排除的节点列表来将该节点从 slurm 作业中排除，方法如下：

```
sbatch --exclude=r7i5n2,r7i5n6 ...
```
but we currently have no way to identify which node is faulty. I think if we switch to pt-1.9.0 or higher where torch elastic replaces the usual launcher. Or we have to use dedicated log files per node via: `#SBATCH --output=%x-%j-%N.out`.

但目前我们无法确定哪个节点出现故障。我认为如果我们切换到 pt-1.9.0 或更高版本，其中 torch elastic 取代了常规的启动器，或者我们可以通过以下方式使用每个节点的专用日志文件：#SBATCH --output=%x-%j-%N.out。

## Watching the training logs(查看训练日志)

On JZ:
```
tail -f $six_ALL_CCFRSCRATCH/checkpoints/tr1-13B/logs/main_log.txt
```

Outside of JZ:
```
perl -e '$u=shift; $b=0; while(1){($e)=qx[curl -sI $u]=~/content-length: (\d+)/; \
print qx[curl -sr $b-$e -L $u] if $e>$b; $b=$e; sleep 300}' \
https://huggingface.co/bigscience/tr1-13B-logs/resolve/main/main_log.txt
```
Currently the updates happen hourly, so this is a delayed version of `tail -f`.

目前更新是每小时进行一次，因此这是 tail -f 命令的延迟版本。

## CodeCarbon


CodeCarbon wasn't ready until the training was over so we only did an additional 10h run to measure with and the to extrapolate to the whole training.

在训练结束之前，CodeCarbon 尚未准备就绪，因此我们只进行了额外的 10 小时运行以进行测量，并据此进行整个训练的推算。

https://huggingface.co/bigscience/tr1-13B-codecarbon

This set of records captures the startup time and 2499 iterations in 2 records per gpu, since there was also an intermediary checkpoint saved half-way and we flush the CC records on each checkpoint saving.

记录中包含了启动时间和 2499 次迭代的信息。每个 GPU 的记录分为两条，可能是因为在训练过程中保存了一个中间检查点，每次保存检查点时会清除 CodeCarbon（CC）的记录。

The training had 168000 iterations. Therefore multiply the reported data by 67. This would be quite approximate since we were using 16 nodes when doing the ramp up, then 64 and only the last 3 weeks 128 nodes.

训练进行了 168,000 次迭代。因此，将报告的数据乘以 67。这是一个相当近似的值，因为我们在进行逐步增加时使用了 16 个节点，然后是 64 个节点，仅在最后三周使用了 128 个节点。

Caveat emptor: I'm not sure whether CC-reports overlap since each report is per gpu and I think they may be measuring the same thing, other than the gpu itself. So this requires research.

请注意：我不确定 CC-报告 是否存在重叠，因为每个报告都是针对每个 GPU 的，我认为它们可能在除 GPU 本身之外测量了相同的内容。因此，这需要进一步研究。

Each csv file contains a report for a single gpu/process. There are 512 reports.

每个 CSV 文件包含了单个 GPU/进程 的报告。总共有 512 个报告。

## Extras
