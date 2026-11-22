# 从零开始配置：在你的集群上运行我们的仿真实验

英文版（步骤完全相同）：[SETUP.md](SETUP.md)。本指南背后的参考文档是 [README.md](README.md)（英文）。
第一次验证运行还有两份更短的文档：一页清单 [QUICKSTART_zh.md](QUICKSTART_zh.md)，以及给操作你集群的 AI agent 的工作指令 [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md)（英文）。

**适用对象。** 任何有集群账号（或一台带 GPU 的工作站）、要运行本发布中仿真实验的人；运行项目发给你的批次时，你**只把数字发回来，不发代码**。本指南带你从一个全新的账号走到：机器通过验证、跑完第一个批次。
下面每一条命令都在发布之前、在一个全新的文件夹里**照原文逐条运行过**；每一步下面给出了预期输出。耗时是在一台 64 核 AMD 机器 + RTX A6000 上测的，你的机器会不一样。

## 我们给你什么，你自己需要什么

| 我们提供 | 内容 |
|---|---|
| 代码 | 公开的 GitHub 仓库 https://github.com/Exiam6/wuji-floating-release （约 8 MB，不需要账号）。发布标签：`wf-20261122-r6` |
| 一个网盘链接 | 资源 tar 包（不经过 GitHub）：文件夹 `full` 里有 15 个 tar，8.69 GB，除 MANO 和 HUG1（这两样你自己下载）之外的全部资源，另有 `SHA256SUMS`、`ASSET_TERMS.txt`、`README.txt`；只跑 CPU 的子集 `rollout_only` 是其中 4 个 tar（0.32 GB），同级的一个文件夹重复放了它的清单文件；如果你运行项目发给你的批次，另有网盘文件夹 `returns/<你的名字>/<日期>/`，用来上传你的结果 |
| 已发布的批次文件夹 | 要做的工作：第一个是 **verification batch 1**（分几部分，先发 1a），在第 8 步之后以压缩包形式发给你 |

| 你自己需要 | |
|---|---|
| 机器 | Linux x86-64（测试过 Ubuntu 22.04 和 24.04）：带 GPU 的工作站，或带登录节点和 GPU 节点的集群 |
| NVIDIA 驱动 | 支持 CUDA 12.8 的驱动（570 或更新）：`nvidia-smi` 可以看到。固定的软件栈是为 CUDA 12.8 编译的 torch 2.9.1；其他 CUDA 版本没有测试过 |
| Python | 3.10（两个环境都测试过）。3.12 只测试过 CPU 环境 |
| 磁盘 | 解包期间约 27 GB 空闲（tar 包 8.7 GB、解包后的资源 8.7 GB、环境 9 GB），删掉 tar 包之后约 18 GB，再加上你的结果。请用快速的本地盘或 scratch 盘（见"故障排除"） |
| 网络 | 只在安装阶段需要（PyPI、download.pytorch.org、data.pyg.org、github.com、huggingface.co）：登录节点即可。跑批次时不需要联网 |
| MANO 账号 | 免费，在 https://mano.is.tue.mpg.de 注册（第 5 步） |

tar 包，以及你必须看到的大小和 sha256：

```text
file                                  bytes   sha256
core.tar                        140,113,920   d618c33f0f8364d31a3a219fb79fb5a33278dd584a8de10d8e8e4da949dc2dd3
hands.tar                        16,670,720   0fb724f749efc40116f7e8d6b2aa7867ead6f36aeb3547dd945a8c2a2bf09601
dinov2.tar                      346,368,000   7cb905da8bcff10a96c0605df43a8eae805fe33d988c08df7a065c535d53367e
hug2.tar                        241,418,240   df52f406f1418545db86fc9500ef408cc1578d4b534a8e6120749160e818fe6b
dumbbell50k.tar                 241,418,240   b4713a3cd331c432b88bd3df4c0e6ea4b324befca79760693f6969bbe746574c
inputs.tar                       83,773,440   3bcc397104e074160b954b549cd2a292ac06fcff6bf3bd3222055b0b697888e7
licences.tar                         40,960   ed94c56584415462853879d8f398b3c2420c4b12c7bbd4f28415c9b8e932e09c
batches.tar                     147,138,560   f3da3a72e05b31680825e55dea99e5c32addb548f25063d7ae28a79375dc692e
prep_ex299_hug1.tar             183,377,920   50d132dda6be8f9178da790b58a702ec83611266375b9e3c8d3d465db887e3ff
prep_ex299_dumbbell50k.tar      195,747,840   a05476242494a6ac6855c5dc79a01f36ac39d60e76dcad1397d8b81041ce1c6f
prep_ex299_hug2.tar             183,377,920   241cc0b34b513ba701294d0e31097b8a75be97c2d26efaae81198815c4938dc6
prep_ex315_hug2.tar           1,394,216,960   4476a37008d666a0c97696196e42ef6bc34c06643af08e669ae0e39f405975e8
prep_ex324_a.tar              2,430,515,200   954227086d067cf421fc6ab46f399bf784e8a0789c2b53f5a3efe975ab252943
prep_ex324_b.tar                656,885,760   56c8114ab7ece578e670b5f057c7a14ab1542bd0fba262a877d83f320613b688
prep_ex338_a_p2.tar           2,430,515,200   93f4af5af53ca6e99eea95137c102ae5a10be467450a54079a34ae71cb2a9f82
Total 8,691,578,880 bytes (15 tarballs). The folder rollout_only holds core, hands, batches and licences (303,964,160 bytes).
```

## 第 0 步：选一个文件夹，检查机器

所有东西都放在一个文件夹 `BASE` 下面。想换磁盘就改第一行。变量保存在 `env.sh` 里：以后每次打开新终端，先运行 `export BASE=...`（同一个值），再运行 `source "$BASE/env.sh"`。

```bash
export BASE=$HOME/wuji-floating
mkdir -p "$BASE" && cd "$BASE"
cat > "$BASE/env.sh" <<'EOF'
: "${BASE:?export BASE=/path/to/your/folder first}"
export RELEASE=$BASE/release
export ASSETS=$BASE/assets
export WF_ASSETS=$ASSETS
export DOWNLOADS=$BASE/downloads
export ENV=$BASE/venv310
export GPUENV=$BASE/gpuenv
export PYTHONDONTWRITEBYTECODE=1
EOF
source "$BASE/env.sh"
uname -m
python3.10 --version
df -h "$BASE" | tail -n 1
```

预期：`x86_64`、`Python 3.10.x`，以及一行你的磁盘信息，可用空间至少 30 GB。如果出现 `python3.10: command not found`：见"故障排除"。在 GPU 节点上还可以运行 `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`，它会打印类似 `NVIDIA RTX A6000, 570.172.08, 49140 MiB` 的内容。

## 第 1 步：获取代码

仓库是公开的：`git clone` 不需要账号，也不需要任何凭据。

```bash
source "$BASE/env.sh"
git clone -c advice.detachedHead=false --branch wf-20261122-r6 https://github.com/Exiam6/wuji-floating-release.git "$RELEASE"
cd "$RELEASE" && git log -1 --format='%H %s' && sha256sum -c --quiet SHA256SUMS && echo "release intact"
```

预期：先打印发布版本的 commit 哈希和标题，然后打印 `release intact`（没有其他输出）。不要修改这个文件夹里的任何文件（见下面的规则框）。

## 第 2 步：两个 Python 环境

CPU 环境跑 rollout，GPU 环境跑 decode 和 DSRL 训练。两个都从固定版本的 requirements 文件构建。如果你们集群的 Python 3.10 不叫 `python3.10`，请在每个代码块的第一行写上它的完整路径。

```bash
source "$BASE/env.sh"
python3.10 -m venv --without-pip "$ENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$ENV/get-pip.py"
"$ENV/bin/python" "$ENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$ENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/cpu.txt"
"$ENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$ENV/bin/python" -m pip check
rm -rf "$ENV/lib/python3.10/site-packages/cmeel.prefix/include"
```

预期 `pip check` 的最后一行：`No broken requirements found.`（约 2 分钟，1.6 GB）。机器人手协议（`sharpa-float-*`、`wuji1hr-float-*`、`wuji1-float-p1`）还需要再装一个文件，只有批次里用到它们时才需要：`"$ENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/cpu_hands.txt"`。

```bash
source "$BASE/env.sh"
python3.10 -m venv --without-pip "$GPUENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$GPUENV/get-pip.py"
"$GPUENV/bin/python" "$GPUENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$GPUENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/gpu.txt"
"$GPUENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$GPUENV/bin/python" -m pip check
```

预期：`No broken requirements found.`（约 7 分钟，7.4 GB：大部分是 CUDA 库）。在有 GPU 的节点上，`"$GPUENV/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 打印 `2.9.1+cu128 True`；在没有 GPU 的登录节点上打印 `False`，现在没关系。

**用 conda 代替 venv。** 把每个代码块的前三行（`python3.10 -m venv`、`curl`、`get-pip.py`）换成
`conda create -y -p "$ENV" python=3.10 pip`（GPU 环境用 `-p "$GPUENV"`），再加一行 `"$ENV/bin/python" -m pip install --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2`；之后的每一行都不变。

## 第 3 步：下载资源并校验

打开网盘链接，把文件夹 `full` 下载到 `$DOWNLOADS`（浏览器会把文件夹下载成若干个 zip：解压后让 15 个 `.tar` 文件和 `SHA256SUMS` 直接放在 `$DOWNLOADS` 里，或者一个一个文件下载；或者用你自己配置的、指向这个网盘的 rclone remote：
`rclone copy "<你的remote>:<共享文件夹>/full" "$DOWNLOADS" --progress`）。如果你只跑 CPU rollout，只需要从 `full` 里下载 `core.tar`、`hands.tar`、`batches.tar`、`licences.tar` 和 `SHA256SUMS`（0.32 GB；同级文件夹 `rollout_only` 重复放了这个子集的清单文件），并用 `sha256sum -c --ignore-missing SHA256SUMS` 校验。
如果集群登录节点连不上网盘，就在自己的电脑上下载，再用 `rsync -a --partial --progress "$DOWNLOADS/" you@cluster:/path/to/wuji-floating/downloads/` 拷过去。

```bash
source "$BASE/env.sh"
cd "$DOWNLOADS" && ls
sha256sum -c SHA256SUMS
```

预期：15 个 tar 包加上 `ASSET_TERMS.txt`、`README.txt`、`SHA256SUMS`，以及 15 行以 `OK` 结尾的输出（`batches.tar: OK`、`core.tar: OK` ……）。以 `FAILED` 结尾的行表示这个 tar 包损坏了：重新下载那一个文件。请阅读 `ASSET_TERMS.txt`：它写明每个部分的许可证和要求的署名。

## 第 4 步：解包到资源根目录

```bash
source "$BASE/env.sh"
mkdir -p "$ASSETS" && for t in "$DOWNLOADS"/*.tar; do tar -xf "$t" -C "$ASSETS"; done
cd "$ASSETS" && sha256sum -c --quiet "$RELEASE/asset_manifests/SHA256SUMS.all" && echo "assets intact"
```

预期：`assets intact`（约 10 秒到 2 分钟，取决于磁盘）。如果只下载了 rollout-only 那一套，第二条命令加上 `--ignore-missing`。解包之后可以删掉 tar 包。

## 第 5 步：MANO 手模型（你自己下载）

MANO 模型不在 tar 包里：每个使用者自己注册并接受它的许可协议。

1. 在 https://mano.is.tue.mpg.de 免费注册，并接受 MANO 许可协议（非商业研究用途）。
2. 下载 MANO 模型 1.2 版：压缩包 `mano_v1_2.zip`。放进 `$DOWNLOADS`。
3. 运行：

```bash
source "$BASE/env.sh"
unzip -q "$DOWNLOADS/mano_v1_2.zip" -d "$DOWNLOADS"
mkdir -p "$ASSETS/mano/models" && cp "$DOWNLOADS/mano_v1_2/models/MANO_RIGHT.pkl" "$ASSETS/mano/models/"
sha256sum "$ASSETS/mano/models/MANO_RIGHT.pkl"
"$ENV/bin/python" "$RELEASE/code/wf.py" assets verify --parts mano
```

预期：先打印 sha256 一行 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767  .../MANO_RIGHT.pkl`，然后是一份以 `"pass": true` 结尾的 JSON 报告。如果哈希不一样，请停下来告诉我们：换了 MANO 文件，所有结果都会变。

## 第 6 步：链接手模型，获取 HUG 源码和 HUG1 权重

```bash
source "$BASE/env.sh"
"$ENV/bin/python" "$RELEASE/code/wf.py" assets link-hands
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug-source
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug1
```

预期：第一条打印带有 `"action": "linked"` 的 JSON；第二条会在固定的 commit 上克隆公开的 HUG 仓库（需要访问 github.com，只需一次），并打印带有 `"pass": true` 和 `"problems": []` 的 JSON；第三条下载公开的 HUG1 权重（需要访问 huggingface.co，只需一次；它们不在 tar 包里），并打印带有 `"pass": true` 的 JSON。`full` 的 tar 包已经包含 DINOv2 权重；如果缺了这一部分，`wf.py assets fetch-dinov2` 会下载它。

## 第 7 步：验证这台机器

请在你将来要计算的那类节点上做这一步。在 SLURM 上，通过交互式分配来运行，例如 `srun -p <partition> --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash`，然后再运行一遍 `export BASE=...` 和 `source "$BASE/env.sh"`。第一条命令不需要 GPU，第二条需要。

```bash
source "$BASE/env.sh"
nice -n 19 "$ENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L1 --workers 4 --out "$BASE/verify_cpu"
```

预期最后一行：`VERIFY PASS (11 checks, 201.6 s)`（秒数随你的 CPU 变化；4 个 worker 约 3.5 分钟）。在它上面，每一行 `l1_...` 在 AMD Zen 2 或 Zen 3 CPU 上都必须写着 `pass_bitwise ... all fields equal: True`。其他 CPU 可能打印 `labels_only`：那不算通过，见规则框。

```bash
source "$BASE/env.sh"
CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L2,L3 --out "$BASE/verify_gpu"
```

预期最后一行：`VERIFY PASS (12 checks, 31.8 s)`。`L2 decode` 几行写着 `PASS_TOLERANCE max |d params99| ...`：同样六张图的 decode 与我们的结果相差 GPU 舍入误差；小于 1e-3 都算通过（我们在 RTX A6000 上相对 RTX 3080 参考值测到 8.2e-6）。`L3 T4_mini_training ... PASS_TOLERANCE` 把 20 次训练调用与我们的结果比较，容差 2 %。

## 第 8 步：把主机指纹和两份报告发给我们

```bash
source "$BASE/env.sh"
CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" fingerprint --out "$BASE/host_fingerprint.json"
ls "$BASE/verify_cpu" "$BASE/verify_gpu"
```

预期：一份 JSON，里面有你的 CPU 型号、`numerical_host_class`（AMD Zen 2 和 Zen 3 CPU 上是 `amd_zen2_zen3_bitwise`，其他是 `other_tolerance`）、你的 GPU 型号和驱动（会列出节点上的每一块 GPU，不管你用哪一块）；然后列出两个报告文件夹。**请把 `host_fingerprint.json` 和两个报告文件夹里的 `verify_report.json` 发给我们**（上传到网盘文件夹 `returns/<你的名字>/setup/`）。我们确认你的主机之后，会发 verification batch 1。在此之前不要跑任何批次：我们最先需要的，是你在这套固定软件栈下的 GPU 验证报告。

## 第 9 步：运行一个已发布的批次

批次是我们以一个压缩包发给你的文件夹（我们会告诉你去哪里下载；放进 `$DOWNLOADS`）。解压之后它就不会再变。下面把批次叫作 `verification_batch_1`：请换成你收到的压缩包的名字（`verification_batch_1a`、`verification_batch_1b` ……）。

```bash
source "$BASE/env.sh"
mkdir -p "$BASE/batches" "$BASE/results"
tar -xf "$DOWNLOADS/verification_batch_1.tar" -C "$BASE/batches"
export BATCH=$BASE/batches/verification_batch_1
export RESULTS=$BASE/results/verification_batch_1
cat "$BATCH/INSTRUCTIONS.txt"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --cpu-python "$ENV/bin/python" --dry
```

`INSTRUCTIONS.txt` 列出这个批次的各个任务组（一个 group 就是一个实验）。`--dry` 逐步打印计划和命令，什么都不运行；如果批次是为另一个版本的发布签发的，而不是你当前检出的版本，它会拒绝运行。然后用三种方式之一运行。

**A. 一台机器、多块 GPU（最直接的方式）。** 每块 GPU 同一时刻跑一个 group；每个 GPU 步骤都有各自的 `CUDA_VISIBLE_DEVICES`。

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --cpu-python "$ENV/bin/python"
```

一块显存足够的卡上要同时跑两个作业，用 `--gpus 0,0`（一个 DSRL 作业约需 4 GB，一个 decode 约 2 GB）。加上 `--workers N` 可以限制 rollout 步骤的 CPU worker 数（默认：核数除以 GPU 槽数，在独占节点上就是全部核）。请在 `tmux` 或 `nohup` 里运行。

**B. SLURM，每个 group 一个 array 任务。**

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch slurm "$BATCH" --out "$RESULTS" --partition <partition> --gpus-per-task 1 --cpus 8 --time 24:00:00 --mem 64G --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
sbatch "$RESULTS/run_batch.sbatch"
```

按你们集群的需要加上 `--account`、`--qos`、`--constraint`、`--gres gpu:a100:1`（代替 `--gpus-per-task`）、`--array-parallel N` 和 `--setup 'module load cuda/12.8'`；提交之前请先读一遍脚本里的 `#SBATCH` 几行。`squeue -u $USER` 可以看到这个 array。

**C. 没有 SLURM（普通 shell 循环）。**

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch shell "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
nohup bash "$RESULTS/run_batch.sh" > "$RESULTS/run_batch.log" 2>&1 &
```

**查看、续跑、回传。** `batch status` 显示哪些 group 已经完成。如果作业中途死掉（超时、节点故障、Ctrl-C），把同一条命令再运行一遍：已经完成的步骤会被跳过，没完成的步骤的文件夹会被移到 `$RESULTS/_failed/`，这个步骤重新运行。用 SLURM 时，`batch status ... --array` 会打印还要运行的 group 编号：`sbatch --array=<那个列表> "$RESULTS/run_batch.sbatch"`。

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch pack "$BATCH" --out "$RESULTS" --note "cluster name, partition, anything unusual"
```

`pack` 会检查每个 group 都已完成，收集我们需要的东西（结果、带已知答案作业的 jobs 文件、步骤戳、训练日志；不含 600 MB 的检查点），写出一个压缩包 `${RESULTS}_pack.tar` 并打印它的 sha256。**请把这一个压缩包上传到网盘文件夹的 `returns/<你的名字>/<YYYYMMDD>/` 里，并把压缩包的名字和 sha256 发消息告诉我们。** 我们检查这些数字（README 第 9 节），在发下一个批次之前告诉你结论。

文件在哪里：`$RESULTS/<group>/<step>/` 里是每个步骤的输出和它的 `STAMP.json`；`$RESULTS/<group>/logs/` 是日志；一个反复失败的 group 会把错误写在 `logs/<step>.err` 和 `<step>.failed.json` 里：请把那个文件夹发给我们。

## 第 10 步：第一个正式作业：verification batch 1

Verification batch 1 复现我们最重要的实验，这样在你接新工作之前，我们可以先拿你的集群和我们的结果对照。它分几部分，每部分是一个压缩包，用第 9 步的方式（A、B 或 C）运行、打包、上传，然后告诉我们：

- **1a** 复现我们的哑铃 census 表（EX331）：在你的 GPU 上对三个抓取先验 decode 164 个留出视角，再在你的 CPU 上用五种手腕条件做 rollout，得到 15 个数字，与我们的数字比较。我们的估算：约 0.1 GPU 小时、13 个 CPU 核小时。我们在你完成第 8 步、并且我们自己的对照运行与已登记的数字一致之后发出。
- **1b** 训练七个 DSRL-NA 种子（在 RTX A6000 这类卡上约 1.3 到 2 GPU 小时），第一个 group 会重复一个我们已经存好的训练，用来拿你的 GPU 和它对照。

每个压缩包里的 `INSTRUCTIONS.txt` 列出它的各个 group。我们会在发下一部分之前告诉你每次回传的结论。

## 规则（一个框）

> 1. **不要修改** 发布文件夹、资源或已发布批次里的任何文件。任何修改都会让 `rollout run` 拒绝启动，并改变每个结果上盖的摘要。
> 2. 如果 `verify` 失败，或者打印 `labels_only` 或 `FAIL`，**请停下**，把报告文件夹发给我们。不要在这台机器上跑批次。
> 3. 告诉我们你的 **CPU 型号和 GPU 型号**：`host_fingerprint.json`（第 8 步）就是做这件事的；换机器、换驱动、换 Python 之后请发一份新的。
> 4. **不要把不同机器的结果合进同一个包。** 一个包对应一种机器。第二个集群有自己的指纹和自己的包。
> 5. 只发数字：就是 `batch pack` 写出的那个压缩包。不发代码、不发检查点、不发 MANO 文件、不发任何凭据。

## 故障排除

| 你看到的现象 | 怎么办 |
|---|---|
| `python3.10: command not found` | 装一个 CPython 3.10（conda、pyenv 或集群的 module），在第 2 步里把完整路径写在 `python3.10` 的位置 |
| `git clone` 连不上（节点不能访问 github.com） | 在另一台机器上克隆，再用 `rsync -a` 把整个文件夹（包括 `.git`）拷到 `$RELEASE`；这个文件夹就是同一个发布版本（第 7 步的 L0 会检查它） |
| `wf: the MANO model is missing`（或 `MANO not linked`） | 第 5 步：文件必须在 `$ASSETS/mano/models/MANO_RIGHT.pkl`；然后重新运行 `assets link-hands` / `fetch-hug-source` |
| `assets verify --parts mano` 说 sha256 不一样 | 你的文件是另一个版本：重新下载 `mano_v1_2.zip`（右手模型，1.2 版）；仍然不一样就停下来告诉我们 |
| `L0 release folder: FAIL ... not in SHA256SUMS` | `$RELEASE` 里有一个多余文件（`__pycache__`、编辑器备份、你自己的笔记）：删掉它，或者 `git status` 再 `git checkout .`；`.git` 文件夹本身是允许的 |
| `import torch` 或者任何一次 `python` 启动要好几分钟 | 环境或资源放在网络文件系统上（NFS、Lustre 的 home）：把 `$ENV`、`$GPUENV` 和 `$ASSETS` 放到节点本地盘或 scratch 盘；我们在 NFS 上测到过 10 到 40 分钟的 import，本地盘上是 3 秒 |
| `CUDA out of memory` | 一张卡一个作业：`--gpus` 里少放几个槽；显存不到 8 GB 的卡跑不了 decode；一个 DSRL 作业约需 4 GB，一个 decode 约 2 GB，再加上别人的作业 |
| `no CUDA device is visible` | 这个节点没有 GPU，或者 `CUDA_VISIBLE_DEVICES` 为空；在 SLURM 上请申请 `--gres=gpu:1` |
| decode 或训练和我们的参考结果只在最后几位小数上不同 | 不同 GPU 型号和驱动版本之间这是正常的（README 第 2 节）。我们的校验程序接受这样的 decode：至少 99.5 % 的行与我们存好的 decode 相差不超过 1e-3，并且所有行都不超过 0.1：在 RTX A6000 上对照我们 RTX 3080 的 decode，HUG1 和 HUG2 没有任何行超过 1e-3，dumbbell 50k 在 7,040 行里有 2 行超过（落在流模型分支附近的样本，最大 4.5e-2）。DSRL 训练日志逐行比较（同一种 GPU 型号上逐位相同，另一种型号上前 100 次调用在 2 % 之内）。一个实验是否“复现”，按每个格子的成功率差（0.31 个百分点算复现，1.0 算接近）和翻转的标签数来判定，不由这些容差决定。超过容差的差别：把报告发给我们 |
| `this batch cannot run on this copy of the release` | 你检出的不是这个批次签发时的版本：`git fetch --tags && git checkout <INSTRUCTIONS.txt 里写的 tag>`，然后把第 7 步重做一次 |
| `the batch folder differs from its SHA256SUMS` | 压缩包损坏或被改过：从原始压缩包重新解压 |
| `sha256sum -c` 对某个 tar 包写 `FAILED` | 重新下载那个 tar 包 |
| `group ... is being run by another process` | 另一个 `batch run` 正在处理这个 group，或者之前的运行在另一个节点上死掉了：删除 `$RESULTS/<group>/RUNNING.lock`，或者加 `--force-unlock` |
| SLURM：`Invalid generic resource (gres) specification` | 你们集群给 GPU 起的名字不同：在 `batch slurm` 里用 `--gres gpu:<型号>:1` 代替 `--gpus-per-task` |
