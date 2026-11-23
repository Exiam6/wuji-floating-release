# 验证运行：你的清单

一页，不做解释。每条命令的预期输出和出错处理见 [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md)（下面的章节号指它；可以交给你的 agent）。完整指南：[SETUP.md](SETUP.md)（英文）和 [SETUP_zh.md](SETUP_zh.md)。英文版：[QUICKSTART.md](QUICKSTART.md)。

**你会收到：**(a) 公开的 GitHub 仓库 `Exiam6/wuji-floating-release`（tag `wf-20261123-r7a`，不需要账号）；(b) 一个网盘链接：文件夹 `full/`（18 个 tar 包，23.3 GB，附 `SHA256SUMS`），如果你为项目运行批次，还有你的上传文件夹 `returns/<你的名字>/`；(c) 发给你的批次：它的压缩包（第一个是 `verification_batch_1a.tar`）及其 sha256（第 5 步之后发给你）。第 1 到 5 步安装并验证发布；第 6 步只用于发给你的批次。
**需要：**Linux x86-64、NVIDIA 驱动 570 或更新、CPython 3.10、本地盘或 scratch 盘上 30 GB 空闲、安装时能上网。设置 `export BASE=<你的文件夹>` 并创建 `env.sh`（AGENT 第 2 节）。

1. **按 tag 克隆。**
   `git clone --branch wf-20261123-r7a https://github.com/Exiam6/wuji-floating-release.git "$RELEASE"`
   `cd "$RELEASE" && sha256sum -c --quiet SHA256SUMS && echo "release intact"`
2. **把 `full/` 下载到 `$DOWNLOADS`，校验，解包**（15 行 `OK`，然后 `assets intact`）。
   `cd "$DOWNLOADS" && sha256sum -c SHA256SUMS`
   `mkdir -p "$ASSETS" && for t in "$DOWNLOADS"/*.tar; do tar -xf "$t" -C "$ASSETS"; done`
   `cd "$ASSETS" && sha256sum -c --quiet "$RELEASE/asset_manifests/SHA256SUMS.all" && echo "assets intact"`
3. **MANO。**在 https://mano.is.tue.mpg.de 注册，把 `mano_v1_2.zip` 下载到 `$DOWNLOADS`。
   `unzip -q "$DOWNLOADS/mano_v1_2.zip" -d "$DOWNLOADS"`
   `mkdir -p "$ASSETS/mano/models" && cp "$DOWNLOADS/mano_v1_2/models/MANO_RIGHT.pkl" "$ASSETS/mano/models/"`
   `sha256sum "$ASSETS/mano/models/MANO_RIGHT.pkl"` 必须输出 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`。
4. **建两个环境**：CPU 环境（`$ENV`）用 `requirements/cpu.txt`，GPU 环境（`$GPUENV`）用 `requirements/gpu.txt`，见 AGENT 第 6 节；两个都以 `No broken requirements found.` 结束。然后运行 `"$ENV/bin/python" "$RELEASE/code/wf.py" assets link-hands` 、`"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug-source` 和 `"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug1`（同一节；HUG1 权重不在 tar 包里）。
5. **验证、指纹、发送、等我们的 go**（AGENT 第 7 到 9 节）。
   `nice -n 19 "$ENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L1 --workers 4 --out "$BASE/verify_cpu"`
   `CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L2,L3 --out "$BASE/verify_gpu"`
   `CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" fingerprint --out "$BASE/host_fingerprint.json"`
   两个 verify 都必须以 `VERIFY PASS` 结束。把下表第一行的文件上传到 `returns/<你的名字>/setup/`。
6. **运行批次 1a、打包、返回**（AGENT 第 10 到 14 节）。
   `tar -xf "$DOWNLOADS/verification_batch_1a.tar" -C "$BASE/batches"`，然后 `export BATCH=$BASE/batches/verification_batch_1a RESULTS=$BASE/results/verification_batch_1a`
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --cpu-python "$ENV/bin/python"`（SLURM：AGENT 第 11 节方式 B；中断后原样再运行同一条命令）
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS"` 必须显示 `"done"` 等于 `"of"`。
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch pack "$BATCH" --out "$RESULTS" --note "集群、分区、任何异常"` 会打印压缩包的 sha256。

| 发回什么 | 文件 | 放哪里（Drive） |
|---|---|---|
| 主机指纹、两份 verify 报告 | `host_fingerprint.json`、`verify_cpu/verify_report.json`、`verify_gpu/verify_report.json` | `returns/<你的名字>/setup/` |
| 批次结果（只有数字） | `${RESULTS}_pack.tar`，另发一条消息写明它的 sha256、你的 CPU 型号和 GPU 型号 | `returns/<你的名字>/<YYYYMMDD>/` |
| 任何失败 | 整个报告文件夹、`$RESULTS/<group>/logs/`、失败命令的输出 | `returns/<你的名字>/failed/<YYYYMMDD>/` |

> **规则。**1. 不改发布文件夹、资源和批次里的任何文件：不改代码，不改资源。2. verify 失败、出现 `labels_only`、`FAIL` 或哈希不一致：**停下，把报告发给我们**；这台机器不要跑批次。3. 一个包一台机器：同一种 CPU 系列和同一种 GPU 型号；不要混用机器，也不要把结果文件夹拿到别处接着跑。4. 告诉我们 CPU 型号和 GPU 型号（指纹文件已包含）。5. 只发数字：只发那个包，不发代码、checkpoint、MANO 文件或任何凭据。各资源部分有各自的许可证（`ASSET_TERMS.txt`）；MANO 文件受你自己的许可协议约束，不得转交他人。6. 我们说 go 之前不跑批次；除非我们要求，不重跑已完成的任务。
