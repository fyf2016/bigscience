# OSCAR


## Megatron pre-processed files(Megatron 预处理文件)

These are the megatron-ready OSCAR files:

这些是准备好的 Megatron OSCAR 文件。

- Full 300M version (529GB) : `$six_ALL_CCFRWORK/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k`

Each folder contains: `meg-gpt2_text_document.bin` and `meg-gpt2_text_document.idx` and Megatron-LM training script expects the following argument:

每个文件夹包含以下内容：`meg-gpt2_text_document.bin` 和 `meg-gpt2_text_document.idx`。而 Megatron-LM 训练脚本需要以下参数：

```
--data-path $six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document
```

Should something get corrupted there is a backup:

如果出现数据损坏的情况，将会有备份可供使用

- Full 300M version (529GB) : `$six_ALL_CCFRSTORE/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRSTORE/datasets-custom/oscar-en-10k`




## How pre-processing was done(预处理是如何完成的)

Here we used the original OSCAR 2019 release: https://oscar-project.org/post/oscar-2019/

在这里我们使用的是原始的 OSCAR 2019 版本：https://oscar-project.org/post/oscar-2019/

In general the process is to first generate jsonl version of the dataset, while filtering out entries smaller than 1K, and then run that jsonl data through Megatron-LM preprocessing tool.

通常的处理流程是首先生成数据集的 JSONL 版本，同时过滤掉小于 1K 的条目，然后将该 JSONL 数据通过 Megatron-LM 的预处理工具进行处理。



The rest of this document is the step by step process of accomplishing that in an efficient way.

这份文件的其余部分是以高效方式完成该目标的逐步过程。

**Update: Now that we better understand Megatron-LM's dataloader we know that it contacts all docs on the fly and delivers seqlen at a time as a single sample ([reference](https://github.com/NVIDIA/Megatron-LM/blob/90e0a0dd08159e1c95f4f9d99bb8687f327d36c3/megatron/data/gpt_dataset.py#L169-L185). So we don't need to filter out docs that are shorter than seqlen. Therefore in the future runs. We should adjust `oscar-to-jsonl.py` to remove the filtering.**

更新：现在我们更好地理解 Megatron-LM 的数据加载器，我们知道它会动态联系所有文档，并且按照 seqlen 大小逐个交付作为单个样本（参考链接）。因此，我们不需要过滤掉长度小于 seqlen 的文档。因此，在未来的运行中，我们应该调整 oscar-to-jsonl.py 以移除过滤操作。

1. Convert `datasets` to `jsonl` which is the format required by Megatron-LM

   将 datasets 转换为 Megatron-LM 所要求的 jsonl 格式。

The main script is [oscar-to-jsonl.py](./oscar-to-jsonl.py). Edit to change languages to use, initially using just English.

主要脚本是 oscar-to-jsonl.py。您可以编辑此脚本以更改要使用的语言，初始情况下仅使用英语。

Note, that since shuffling slows the writeout process by 5-7 times, we don't shuffle in the script, but post-process it externally. See step 3.

请注意，由于 shuffling 操作会使写入过程变慢 5-7 倍，因此我们在脚本中不进行 shuffling，而是在外部进行后处理。请参考第 3 步骤。

To launch: [oscar-to-jsonl.slurm](./oscar-to-jsonl.slurm).
启动脚本：oscar-to-jsonl.slurm。

With "unshuffled_deduplicated_en" after filtering large entries (`>=1024`) we end up with 70754K examples out of 304230K total (about 1/4th of the full dataset).

经过对大型条目（>=1024）进行过滤后，“unshuffled_deduplicated_en” 结果为 70754K 个示例，占 304230K 总数的约 1/4。

The result is 5 files `oscar-[0-4].jsonl` of about 250GB each.

结果将被分成 5 个文件 oscar-[0-4].jsonl，每个文件约 250GB。

Runtime: 2-3h to download, ~2h to build, ~8h to filter, ~1.5h to write shards out

运行时间：下载数据需要 2-3 小时，构建数据需要约 2 小时，过滤数据需要约 8 小时，写出分片数据需要约 1.5 小时。

Update: `datasets` added multiproc `to_json` support:

更新：datasets 已添加了多进程的 to_json 支持：

https://github.com/huggingface/datasets/pull/2747

so it is in master, or after next after 1.11 version is released.

因此，它将在主分支中出现，或者在发布 1.11 版本之后的下一个版本中可用。


2. Concatenate(拼接)

```
cat oscar-[0-4].jsonl > oscar-en.jsonl
```

This gives us a 1.2TB file.

这将会生成一个 1.2TB 的文件。

Check:
```
$ wc -l oscar-en.jsonl
304230423 oscar-en.jsonl
```

Runtime: a few minutes


3. Shuffle

Megatron requires users to do their own shuffling of jsonl input.

Megatron 要求用户自行对 jsonl 输入进行洗牌。

It was too slow to do inside the filtering script, so we are using a post-processing solution.

在过滤脚本内部执行洗牌操作速度过慢，因此我们使用了后处理的解决方案。

Using https://github.com/alexandres/terashuf and 150GB RAM in ~1.5h we shuffle the file.

使用 https://github.com/alexandres/terashuf 和大约 150GB 的内存，我们可以在大约 1.5 小时内对文件进行洗牌。

Important: note that the slurm job uses SCRATCH for `TMPDIR` and also sets the memory limit it can use to 150.0 (GB) (slightly under 160GB available on this slurm allocation to allow for other processes).

重要提示：请注意，Slurm 作业使用 SCRATCH 作为 TMPDIR，并将其可使用的内存限制设置为 150.0（GB）（略低于此 Slurm 分配中可用的 160GB，以便为其他进程留出空间）。

To launch: [oscar-fast-shuffle.slurm](./oscar-fast-shuffle.slurm)

启动脚本：[oscar-fast-shuffle.slurm](./oscar-fast-shuffle.slurm)

`terashuf` is in `$six_ALL_CCFRWORK/bin/terashuf`

terashuf 位于 $six_ALL_CCFRWORK/bin/terashuf

The result is `oscar-shuffled.jsonl`

结果为 oscar-shuffled.jsonl

Runtime: 2h

运行时间：2 小时

4. Megatron-LM preprocess（Megatron-LM 预处理）

**Update**: that was an error, we can actually run for 100h on `-p cpu_p1` and so the normal script can complete no problem, but as a result of this mistake we can now pre-process data much faster.

更新: 这是一个错误，实际上我们可以在 -p cpu_p1 上运行 100 小时，所以正常的脚本可以顺利完成，但由于这个错误，我们现在可以更快地预处理数据。

We only have 20h to do processing which is not enough to process 300M records. Trying to do the whole thing in one preprocessing script took more than 24h and thus failed. Adding more than 16 workers didn't speed things up.

我们只有 20 小时来进行处理，这不足以处理 3 亿条记录。尝试在一个预处理脚本中完成整个过程超过了 24 小时，因此失败了。增加 16 个以上的工作节点并没有加快处理速度。

So we are splitting it in 4 chunks of ~80M records

因此，我们将其分成 4 个约 8000 万条记录的块进行处理。

```
split -l 77000000 oscar-en-shuffled.jsonl oscar
mv oscaraa oscar-en-shuffled-p1.jsonl
mv oscarab oscar-en-shuffled-p2.jsonl
mv oscarac oscar-en-shuffled-p3.jsonl
mv oscarad oscar-en-shuffled-p4.jsonl
```

We do the pre-processing:

我们进行预处理：

The main script to launch: [oscar-jsonl-to-meg-gpt2.slurm](./oscar-jsonl-to-meg-gpt2.slurm), and we need to make copies of it for each chunk:

启动主要脚本：[oscar-jsonl-to-meg-gpt2.slurm](./oscar-jsonl-to-meg-gpt2.slurm)，我们需要为每个块制作副本：

```
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-1.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-2.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-3.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-4.slurm
perl -pi -e 's|p1|p1|' oscar-jsonl-to-meg-gpt2-1.slurm
perl -pi -e 's|p1|p2|' oscar-jsonl-to-meg-gpt2-2.slurm
perl -pi -e 's|p1|p3|' oscar-jsonl-to-meg-gpt2-3.slurm
perl -pi -e 's|p1|p4|' oscar-jsonl-to-meg-gpt2-4.slurm
```

```
sbatch oscar-jsonl-to-meg-gpt2-1.slurm
sbatch oscar-jsonl-to-meg-gpt2-2.slurm
sbatch oscar-jsonl-to-meg-gpt2-3.slurm
sbatch oscar-jsonl-to-meg-gpt2-4.slurm
```

This took about 6h each but run in parallel on different instances. This is surprisingly the projected time for the initial attempt to run in in one chunk, which was projected to 24 hours, and couldn't fit into 20h cap. So we finished the whole thing in 6 hours.

每个块的处理时间大约为 6 小时，但是在不同实例上并行运行。这令人惊讶地与最初尝试在一个块中运行并预计需要 24 小时相符，而无法在 20 小时内完成。所以我们在 6 小时内完成了整个过程。

Outcome:

结果

```
$ ls -1sh meg-gpt2-p*
131G meg-gpt2-p1_text_document.bin
1.4G meg-gpt2-p1_text_document.idx
131G meg-gpt2-p2_text_document.bin
1.4G meg-gpt2-p2_text_document.idx
131G meg-gpt2-p3_text_document.bin
1.4G meg-gpt2-p3_text_document.idx
138G meg-gpt2-p4_text_document.bin
1.5G meg-gpt2-p4_text_document.idx
```

Next merging: [oscar-meg-gpt2-merge.slurm](./oscar-meg-gpt2-merge.slurm)

接下来是合并操作：[oscar-meg-gpt2-merge.slurm](./oscar-meg-gpt2-merge.slurm)

Runtime: 22min - needed 26GB RSS RAM

运行时间：22分钟 - 需要 26GB RSS RAM

Outcome: 304_230_423 records

结果：304,230,423 条记录

```
$ ls -1sh meg-gpt2_text_document.*
529G meg-gpt2_text_document.bin
5.7G meg-gpt2_text_document.idx
```

Total runtime: under 7h.

总运行时间：不到 7 小时。

Let's also make a small 10k version for experiments:

我们还可以制作一个用于实验的小型版本，包含 10,000 条记录。

```
head -10000 oscar-shuffled.jsonl > oscar-shuffled-10k.jsonl
```
and then process with the same slurm script above, but changing the input to `oscar-shuffled-10k.jsonl`

然后，使用上面相同的 Slurm 脚本进行处理，但将输入更改为 oscar-shuffled-10k.jsonl。


5. Final destination（最终目标）

We did all the processing on the SCRATCH partition which gets wiped out every 30 days, so we need to move the files to where they will not be deleted.

我们在 SCRATCH 分区上进行了所有处理，该分区每 30 天会被清空，因此我们需要将文件移动到不会被删除的位置。

Since at this moment we used just the English part of the OSCAR dataset, let's include that in the folder name to differentiate from other builds that will be multi-lingual.

由于目前我们只使用了 OSCAR 数据集中的英语部分，请在文件夹名称中包含这一信息，以便与其他多语言版本的构建区分开来。

Make the final result which will be used by the megatron training script available on the persistent WORK partition:

将最终结果移动到持久工作分区上，以供 megatron 训练脚本使用：

```
mkdir oscar-en
mv meg-gpt2_text_document.* oscar-en
cp -r oscar-en $six_ALL_CCFRWORK/datasets-custom
```

Back it up to STORE:

将其备份到 STORE：

It's already binary and just 2 files, so no need to tar (STORE has limited inodes)

如果数据已经是二进制格式且只有两个文件，那么您无需将它们打包成 tar 文件（STORE 分区的 inode 数量有限）。

```
mkdir -p $six_ALL_CCFRSTORE/datasets-custom
cp -r oscar-en $six_ALL_CCFRSTORE/datasets-custom
```

Also copy the small version for experiments to WORK and STORE:
还需要将用于实验的小型版本复制到 WORK 和 STORE 分区：

```
cp -r oscar-en-10k $six_ALL_CCFRWORK/datasets-custom
cp -r oscar-en-10k $six_ALL_CCFRSTORE/datasets-custom
```

Tar/gz `oscar-shuffled.jsonl` and the dataset files to STORE, using [oscar-to-backup-tgz.slurm](./oscar-to-backup-tgz.slurm):

使用 oscar-to-backup-tgz.slurm 脚本，将 oscar-shuffled.jsonl 和数据集文件打包成 tar.gz 格式，并将其备份到 STORE 分区。

```
sbatch oscar-to-backup-tgz.slurm
```

6. Estimate total number of tokens（评估 tokens 的总数）

Make a 1GB slice:
```
$ head -79000 oscar-en-shuffled.jsonl > oscar-1GB.jsonl
$ ls -sh oscar-1GB.jsonl
1.0G oscar-1GB.jsonl
```

Analyze it (low mem-footprint):
```
$ python -c "import json, sys; \
from transformers import GPT2TokenizerFast; \
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2'); \
print(sum(tokenizer(json.loads(l)['text'], return_length=True).length[0] for l in sys.stdin.readlines()))" < oscar-1GB.jsonl
234260484
```

Extrapolate:

推算：

Thus 234M tokens in 1GB, ~280B tokens in 1.2TB (`234*1200`)

根据计算，1GB 中大约有 234M 个标记，因此在 1.2TB（234*1200）中大约有 280B 个标记。

Incidentally this coincides with @Yozh's `FILE_SIZE_IN_GBS/4.5` formula! (average 4.5chars per word)

有趣的是，这与 @Yozh 提到的 FILE_SIZE_IN_GBS/4.5 公式相吻合！该公式基于平均每个单词包含 4.5 个字符进行估算。


