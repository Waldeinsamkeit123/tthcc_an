# AGENTS Guide for `tthcc_an`

## 目标

这个仓库用于 Run 3 boosted `ttHcc` / `ttHbb` study，目前并行维护三条主分析线：

- boosted Higgs tagger study
- prototype event-level BDT study
- event-level multiclass NN output study

其中第一条线的核心工作是从 Pepper 产出的 ROOT ntuple 中读取 `Events` tree，构建 AK8 fatjet 级别样本，计算 working points / ROC / 区域效率，并输出表格、JSON 摘要和 CMS 风格图。第二条线目前聚焦于 `0L` 的事件级 BDT 原型。第三条线目前研究 2024 boosted `1L` 的事件级 multiclass NN 输出，后续用于扩展 `0L`、模型比较和 mass-sculpting study。

后续代理在这个仓库里工作时，默认目标应当是：

- 先理解并保留现有分析约定，而不是重写工作流。
- 优先通过 `config/*.json` 调参，只有在逻辑本身需要变化时才改 `src/`。
- 保持 CLI 行为、输出文件名、payload schema 尽量稳定。
- 默认使用中文与用户沟通；除非用户明确要求英文，否则说明、进度更新、结果总结都使用中文。

## 快速上手

推荐环境：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
```

最小 Python 依赖写在 `requirements.txt`：

- `numpy`
- `uproot`
- `awkward`
- `matplotlib`
- `mplhep`
- `scikit-learn`（event-BDT prototype，通常由 `LCG108` 提供）
- `xgboost`（event-BDT prototype，通常由 `LCG108` 提供）

主入口：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json
```

当前 shipping 的 2024 boosted-study config 指向：

- ntuple 目录：`/eos/user/h/hanw/ttHcc/pepper_data/2024/add_nonttbarmatch_event`
- `gen_sumw`：`/eos/user/h/hanw/ttHcc/pepper_data/2024/add_nonttbarmatch/gen_sumws.json`

若用户要重做 `gpart_higgs_vs_qcd` vs `gpart_xbb_vs_xcc` 的 `QCD mistag eff = 0.1% / 0.5%` 研究，记得显式传：

```bash
--xbb-vs-xcc-region-preset tight
```

如果只是调整 contour 标注位置、图例或 CMS style，优先使用已有 cache 做 `--plot-only` 重画。当前常用缓存是：

- `outputs/boosted_higgs_tagger_study_2024_tight_20260525/plot_input.npz`

对应命令：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-only \
  --plot-input outputs/boosted_higgs_tagger_study_2024_tight_20260525/plot_input.npz \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight_20260525 \
  --xbb-vs-xcc-region-preset tight
```

常用命令：

- smoke test：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --max-files-per-sample 1 \
  --xbb-vs-xcc-region-preset tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_smoke_tight
```

- 本地完整重跑：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --xbb-vs-xcc-region-preset tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight
```

- chunked Condor 重跑：

```bash
python scripts/submit_boosted_higgs_tagger_condor.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --workflow-mode chunked \
  --files-per-chunk 20 \
  --condor-dir condor/boosted_higgs_tagger_2024_tight \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight \
  --submit \
  -- --xbb-vs-xcc-region-preset tight
```

event-BDT 原型入口：

```bash
python scripts/run_event_bdt.py \
  train \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json
```

event-level NN study 入口：

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json
```

推荐 smoke test：

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --max-files-per-sample 1 \
  --outdir /tmp/nn_study_2024_1l_v1_smoke
```

常用辅助入口：

- `scripts/submit_boosted_higgs_tagger_condor.py`: 生成并可提交 HTCondor workflow
- `scripts/run_boosted_higgs_tagger_study.py`: repo 根目录下的薄包装脚本，真正逻辑在 `src/tthcc_an/boosted_higgs_tagger_study.py`
- `scripts/run_event_bdt.py`: event-BDT prototype CLI
- `scripts/run_nn_study.py`: event-level multiclass NN score-study CLI

## 仓库结构

- `config/`
  - 分析配置 JSON
  - 正常分析调参应优先改这里
  - 支持 `#` 和 `//` 行注释，解析逻辑在 `config_loader.py`
  - `config/event_bdt/` 里是 event-BDT 样本配置和训练配置
  - `config/nn_study/` 里是 event-level NN study 配置
- `src/tthcc_an/`
  - 核心分析代码
  - `src/tthcc_an/event_bdt/` 里是 event-BDT 原型模块
  - `src/tthcc_an/nn_study/` 里是独立的 NN score-study 子系统
- `scripts/`
  - 本地运行和 Condor 提交脚本
- `outputs/`
  - 本地产出目录，通常是生成物
- `condor/`
  - Condor workflow、chunk manifest、chunk 输出、日志等，通常是生成物
- `crosssections_run3.json`
  - xsec 输入

## 核心模块职责

### `src/tthcc_an/boosted_higgs_tagger_study.py`

这是主控文件，负责：

- 解析 CLI 参数
- 从 config 和 CLI 合并有效参数
- 读取 ROOT 文件
- 选择 AK8 jets
- 计算各类 score
- 驱动 working point / ROC / plot / summary 输出
- 处理三种主要运行模式：
  - 直接读 ROOT 做完整分析
  - `--export-chunk` 导出 NPZ chunk payload
  - `--merge-chunks` 合并 chunk payload
  - `--plot-only` 只重画图

如果一个改动影响“整体流程”或新增 CLI 选项，优先看这里。

### `src/tthcc_an/config_loader.py`

负责：

- 读带注释 JSON
- 展开样本文件 glob
- 解析 `study` / `plot` / `normalization`
- 提供默认值
- 解析输出目录

如果需求只是：

- 改默认参数
- 增加新的 config 字段
- 调整配置优先级

优先改这里和相应 README / AGENTS 描述。

### `src/tthcc_an/definitions.py`

集中放“分析字典”和常量：

- `FATJET_FIELDS`
- truth label 定义与顺序
- process 显示顺序与颜色
- target 定义
- score 标签
- score 所需输入字段
- contour plot preset 和 region 定义

如果新增 score、truth category、process 样式或 contour preset，这里通常是第一站。

### `src/tthcc_an/metrics.py`

负责数值计算：

- weighted efficiency / weighted quantile
- working points
- ROC
- histogram payload 下的等价计算

涉及公式、cut 定义、统计量时优先看这里。

### `src/tthcc_an/payload_io.py`

负责 chunk payload I/O：

- `raw_v1`
- `histogram_v1`
- score histogram 打包/合并
- contour histogram 打包/合并

如果改动会影响 `--export-chunk` / `--merge-chunks` / `plot_input.npz` 兼容性，这里必须同步检查。

### `src/tthcc_an/plotting.py`

负责：

- CMS 风格绘图
- score distribution
- ROC
- background process plots
- contour plots
- fixed-x / fixed-other-efficiency scans

绘图样式、图例、坐标、额外诊断图都在这里。当前统一使用 `CMS Simulation` 标注，不再显示 `Private Work`。如果要调 `xbb-vs-xcc` contour 上 `Hbb` / `Hcc` region label 的位置，先看 `definitions.py` 里的 `annotation_x` / `annotation_y` / `ha`，再看这里的 annotation 实现。

### `src/tthcc_an/reporting.py`

负责文本和结构化输出：

- CSV
- JSON
- summary text
- contour / scan text summary

如果输出 schema 或文案有变化，这里通常需要同步。

### `scripts/submit_boosted_higgs_tagger_condor.py`

负责 Condor 工作流生成：

- single-job 模式
- chunked DAG 模式
- chunk manifest 生成
- wrapper shell script
- submit file / DAG file

这是“批处理编排器”，不是物理分析逻辑本体。

### `src/tthcc_an/event_bdt/`

这是 event-level BDT 原型的独立子系统。目前主要包含：

- `config.py`
  - 训练配置和样本配置加载
- `dataset.py`
  - 读取 ROOT、应用 event-level preselection、准备训练缓存
- `reweighting.py`
  - 训练前的背景 shape reweighting
- `training.py`
  - k-fold XGBoost 训练、OOF score、summary 输出，以及 per-fold eval history
- `plotting.py`
  - ROC、score-shape、training curve、threshold scan、ttH-score study、QCD 质量形状比较绘图
- `cli.py`
  - `prepare/train/evaluate/predict` 子命令，以及 scan / study 输出编排

如果一个改动涉及 event-BDT workflow，优先在这个子目录内完成，不要把逻辑揉进 boosted fatjet study 主流程。

### `src/tthcc_an/nn_study/`

这是 event-level multiclass NN 输出研究的独立子系统：

- `config.py`
  - 读取 channel、样本、score/truth 定义、selection、归一化和绘图设置
- `dataset.py`
  - 读取 ROOT、跳过 metadata-only 文件、应用 sample stitching / event selection、构建显式权重
- `definitions.py`
  - 计算配置表达式并检查 truth category 是否重叠
- `metrics.py`
  - weighted confusion、pairwise discriminator、精确 weighted ROC 和标准 AUC
- `plotting.py`
  - score shape/yield、confusion、pairwise ROC 和 AUC matrix
- `reporting.py`
  - JSON/text summary
- `cli.py`
  - 运行编排和 `--max-files-per-sample` / `--only` 支持

当前 shipping config 是 `config/nn_study/nn_study_2024_1l_v1.json`。涉及 NN study 的改动应留在这个子系统，不要混入 boosted tagger 或 event-BDT。

## 推荐改动路径

### 只是调样本、luminosity、默认 target/score/cut

先改：

- `config/*.json`

其次才考虑：

- `config_loader.py`
- `README.md`

### 新增一个 score

通常需要同时检查这些点：

1. `definitions.py`
   - `SCORE_LABELS`
   - `SCORE_INPUT_FIELDS`
   - 可能的 target 默认 score
2. `boosted_higgs_tagger_study.py`
   - score 计算逻辑
   - 所需字段读取
3. `plotting.py`
   - 是否支持该 score 的分布图和 ROC
4. `payload_io.py`
   - chunk payload 是否能携带该 score
5. `README.md` / `AGENTS.md`
   - 更新文档

### 改 truth 分类或 target 定义

必须同步检查：

- `definitions.py`
- `metrics.py`
- `reporting.py`
- 任何依赖 truth code 顺序的 histogram / plotting 逻辑

truth label 顺序是 payload 和输出的一部分，不要随意改。

### 改 contour region 或 preset

重点看：

- `definitions.py`
- `plotting.py`
- `reporting.py`
- `config/*.json` 中的 `xbb_vs_xcc_region_preset`

### 改 event-BDT 的样本、feature、预选或权重

通常先改：

- `config/event_bdt/samples_*.json`
- `config/event_bdt/train_*.json`

其次才考虑：

- `src/tthcc_an/event_bdt/dataset.py`
- `src/tthcc_an/event_bdt/reweighting.py`
- `src/tthcc_an/event_bdt/training.py`
- `src/tthcc_an/event_bdt/plotting.py`

目前 event-BDT 原型的设计原则是：

- 样本列表与训练设置分离
- 当前 shipping sample config 是 `config/event_bdt/samples_2024_0l_jecs_v1.json`
- 当前 shipping 训练 config 是 `train_ttHcc_0l_3class_baseline_jecs.json` 和 `train_ttHcc_0l_4class_baseline_jecs.json`
- `3-class` 是主默认工作流；`4-class` 主要用于分开研究 `ttHbb` 和 `ttHcc`
- `eval-only` process 目前是 `ttH_nonbb`，通过 `eval_processes_extra` 和样本 role 控制，不参与训练
- `TargetFatJet_*` 特征默认按每事件第一个对象展开；`CleanedJet_*__N` 默认取按 `pt` 排序后的前几个 jet
- mass study 依赖 `TargetFatJet_msoftdrop`、`TargetFatJet_regressed_mass_generic`、`TargetFatJet_regressed_mass_x2p` 这些 analysis branches；若缺失，需要先 `prepare --force`

## 运行模式和产物

### 直接完整运行

输入：

- ROOT 文件
- config JSON
- normalization JSON

输出通常在 `study.outdir` 指定目录下，包含：

- `tables/*.csv`
- `summaries/*.txt`
- `plots/*`
- `study_summary.json`
- `plot_input.npz`

如果 ROOT ntuple 或 `gen_sumws.json` 更新了，必须重新跑完整分析或 chunked merge 链路。
`--plot-only` 只会消费已有 `plot_input.npz`，不会自动拾取新的输入样本。

### Chunk 导出

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config ... \
  --export-chunk /path/to/chunk.npz
```

默认使用 `histogram` payload 模式，这样 merge 内存更低。除非明确需要 raw-level payload，否则不要轻易改回 `raw`。

### Chunk 合并

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config ... \
  --merge-chunks 'condor/.../chunk_outputs/chunk_*.npz'
```

### 只重画图

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config ... \
  --plot-only \
  --plot-input outputs/.../plot_input.npz
```

适合只改绘图逻辑或 plot style 时快速迭代。

### Event-BDT 原型

event-BDT 原型目前有四个模式：

- `prepare`
  - 读 ROOT 并写出 `prepared_inputs.npz`
- `train`
  - 读 prepared cache，做 k-fold XGBoost 训练，并写模型、OOF 预测、summary
- `evaluate`
  - 读 `predictions.npz`、`prepared_inputs.npz`、`training_summary.json`，重画 ROC、score-shape、training curve、threshold scan 和 `tth_score_study`
- `predict`
  - 读训练好的模型，重写 ROOT 并追加 score 分支

当前原型还没有：

- 专门的 Condor submission helper
- 独立测试套件

`evaluate` 现在不只是简单重画 ROC。只要缓存齐全，它也会补画：

- `plots/training_curve__mlogloss.png`
- `plots/training_curve__merror.png`
- `class_score_threshold_scan.txt/json`
- `qcd_score_threshold_scan.txt/json`
- `plots/score_by_training_class_weighted_events__tth_scores_qcd_drop_<97|98|99>_logy.png`
- `plots/tth_score_study/*`
- `tth_score_study_summary.txt/json`

如果只改了 mass 分析分支或绘图逻辑，优先考虑 `prepare --force` 加 `evaluate`，不要默认重训。

### Event-level NN study

当前 NN study 直接读取 Pepper 最终 `Events` ntuple，不训练模型。默认产物包括：

- 每个 NN 输出的 `score_<class>.png` 和 `score_<class>__yield.png`
- `confusion_matrix_truth.png` 和 `confusion_matrix_pred.png`
- 每个可用 signal truth class 的 `roc_<class>.png`
- `pairwise_auc_matrix.png`
- `nn_study_summary.json` 和 `nn_study_summary.txt`

分类输出放在 config 的 `scores`，其名称和顺序必须与 `truth_categories`
一致，并参与 confusion、pairwise ROC 和 AUC。只需要画 shape/yield、但不对应
独立 truth 类的组合输出放在 `auxiliary_scores`；当前 0L/1L 均配置了
`score_ttH`、`score_ttZ`、`score_ttX`、`score_Xbb`、`score_Xcc` 和
`score_ttjets`。0L 额外把 `score_qcd` 作为正式分类输出，truth 为
`n_gentop == 0`。

`--only scores confusion roc auc mass-sculpting qcd-score-scan` 可限制绘图组，但 summary 总会写出。
`--only mass-sculpting` 会按需只读配置中的 scan score、mass、truth/stitching
和 weight branches，并跳过 score/confusion/ROC/AUC 计算。当前没有
cache/plot-only 或 Condor 支持，因此 mass-only 仍需扫描 ROOT events。
`--only qcd-score-scan` 只读 `score_qcd`、truth/stitching 和 weight 所需分支，
跳过普通 score、confusion、ROC/AUC 和 mass-sculpting。

## 重要分析约定

- 默认候选策略是 `mass_window_all_jets`
- 默认质量窗是 `100 GeV <= msoftdrop <= 150 GeV`
- 默认 AK8 选择是 `pt >= 200 GeV` 且 `|eta| <= 2.4`
- 权重使用：
  - `sample_norm = lumi_fb * xsec / gen_sumw`
  - analysis weight: `sample_norm * abs(weight)`
  - bookkeeping signed weight: `sample_norm * weight`
- config 参数优先级：
  - CLI 显式参数
  - config JSON
  - 内建 fallback
- shipping 的 2024 boosted config 默认 `xbb_vs_xcc_region_preset = loose`
- 若用户要重做 `QCD mistag eff = 0.1% / 0.5%` 的 `xbb-vs-xcc` contour / fixed-other scan，使用 `--xbb-vs-xcc-region-preset tight`
- 当前所有图统一使用 `CMS Simulation` 标注；除非用户明确要求，否则不要恢复 `Private Work`
- 当前 tight contour 常用输出是 `outputs/boosted_higgs_tagger_study_2024_tight_20260525`
- `Hbb` / `Hcc` contour label 现在通过 slightly-left-shifted + `ha='right'` 避免挤出画布；若要继续调位置，看 `definitions.py` 和 `plotting.py`

event-BDT 原型额外约定：

- 当前 shipping sample config 是 `samples_2024_0l_jecs_v1.json`，输入路径是 `/eos/user/h/hanw/ttHcc/pepper_data/2024/JECs_v1_events`
- 当前 active process 主要是 `qcd`, `ttbar`, `ttHbb`, `ttHcc`，以及 `eval_only` 的 `ttH_nonbb`
- 当前 shipping 训练 config 默认使用 `k_folds = 5`
- 当前 `3-class` 训练类是 `tth = ttHbb + ttHcc`, `ttbar = ttbar + ttbb + ttll`, `qcd = qcd`
- 当前 `4-class` 训练类是 `tthbb`, `tthcc`, `ttbar`, `qcd`
- 当前常用新增特征包括 `minDEta_TarClean`, `minDPhi_TarClean`, `CleanedJet_pt__1..4`, `CleanedJet_tag__1..4`
- 当前 mass 分析分支包括 `TargetFatJet_msoftdrop`, `TargetFatJet_regressed_mass_generic`, `TargetFatJet_regressed_mass_x2p`
- `class_score_threshold_scan` 的 keep 方向是：signal-like scores 用 `score >= cut`，只有 `bdt_score_qcd` 用 `score <= cut`
- `qcd_score_threshold_scan` 里的 auxiliary ttH significance 语义是：
  - `3-class`: `S = ttH(bb+cc)`, `B = ttbar + QCD`
  - `4-class`: `S = ttHbb + ttHcc`, `B = ttbar + QCD`
- `tth_score_study` 里 QCD-cut significance 当前定义为：
  - `ttHbb/sqrt(QCD)` 用 `S = ttHbb`, `B = QCD`
  - `ttHcc/sqrt(QCD)` 用 `S = ttHcc`, `B = QCD`
  - 都在 `100 <= TargetFatJet_msoftdrop <= 150 GeV` 窗口内评估
- 当前 `4-class` `tth_score_study` 还会产出 QCD-only fine-binning 质量比较图，包括单 panel 的 `100-150 GeV` overlay

NN study 额外约定：

- 当前 input 是
  `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v1_pred_RealMass_events`
- normalization metadata 是
  `/eos/user/h/hanw/ttHcc/pepper_data/2024/NNeval_{0L,1L}_v1_pred_RealMass/gen_sumws.json`
- 两个 config 使用 `study.sample_file_pattern = {input_location}/{name}/*.root`；
  同目录结构的新 prediction 只需改 `input_location`、`gen_sumw_file` 和 `outdir`，
  不要恢复逐 sample 的绝对 `files` 路径
- 实际 NN 分支使用 `score_<class>`，包含 `score_ttZqq`；12 个 score class 与 truth class 必须同名同序
- truth 分支是 `higgs_decay`, `z_decay`, `n_gentop`, `tt_hf_flavor`, `tt_hf_count`
- truth 定义保持 resolved/Pepper 现行约定；`tt_hf_count` 的 1/2/>2 分别对应 single-extra、double-extra、pair-extra 类，不要静默重定义
- ntuple 已在 Pepper `HasTargetFatJet` 末级写出，且此前已过 `ReqMassWindow`；shipping config 不额外加 `nJet > 4` 或其他 physics cut
- inclusive `TTto*` 只保留 LF/c 类；`TTBBto*` 与 `TTto*-BBDPS` 只保留 b 类，避免 inclusive bottom 重复
- shape plot 按 truth class 分别归一到 1；yield/confusion/ROC/AUC 使用 `sample_norm * abs(weight)`
- pairwise discriminator 是 `score_i / (score_i + score_j)`，零分母取 0.5
- AUC 使用标准 `integral(TPR dFPR)`；图上仍按需求画 `x=signal efficiency`, `y=background efficiency`
- 1L config 当前启用 `TargetFatJet_regressed_mass_generic` mass-sculpting：
  - `score_ttX > 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9`
  - `score_ttLF < 0.01, 0.02, 0.05, 0.1`
- mass config 使用 `populations` 列表；每项可分别通过 `samples` 精确选择
  config sample name，并通过 `truth_categories` 进一步选择 truth 类
- 当前 ttHcc、ttHbb、ttZ 各由三个 decay-channel dataset 组成，mass populations
  通过稳定的 sample label 分别聚合为 `ttHcc_sample`、`ttHbb_sample`、
  `ttZ_sample`；`all_background_samples` 通过 `exclude_sample_labels` 排除这三类
- 0L 除 `score_ttX` 和 `score_ttLF` 外，还扫描
  `score_qcd < 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1`
- 0L 另有配置驱动的 `qcd_score_scan`：实际使用 `score_qcd < cut`，候选点与
  mass-sculpting 相同，并在 `1e-6` 到 `1` 做 fine log scan；不要加入旧脚本的
  `nJet > 5`
- QCD-score scan 的 `tt+X` 固定为
  `ttLF, ttcj, ttcc, tt2c, ttbj, ttbb, tt2b`，QCD 使用现有
  `qcd: n_gentop == 0` truth，因此包含配置样本中满足该
  truth 的 QCD/Wto2Q/Zto2Q 事件
- QCD-score expected yield 和 weighted efficiency 使用
  `sample_norm * abs(weight)`；输出位于 `plots/qcd_score_scan/` 和
  `summaries/qcd_score_scan.{json,txt}`，不自动选择最终 working point
- `qcd_score_distribution.png` 使用 linear `score_qcd` 的 logarithmic bins，
  各组独立归一化；展示组为 `ttHcc`、`ttLF`、合并 charm
  (`ttcj, ttcc, tt2c`)、合并 bottom (`ttbj, ttbb, tt2b`) 和 `qcd`
- 图中的竖线和右侧 rejected 阴影只由
  `qcd_score_scan.reference_threshold` 控制，它只是可视化参考值，不是最终 cut
- mass 范围复用 event-BDT 约定：加权 `0.5%–99.5%` 分位加 4% padding，45 bins
- mass shape 使用 `sample_norm * abs(weight)` 后在绘图区间独立归一到积分 1；
  下方面板显示每个 mass bin 的 `kept / uncut`
- 输出为 `plots/mass_sculpting__<mass>__<score_branch>.png` 和
  `plots/mass_sculpting__<mass>__<score_branch>__<population>.png`（默认
  `all_selected_mc` 省略 population suffix），以及
  `summaries/mass_sculpting_summary.{json,txt}`；未来 0L 只应扩 config，不复制脚本

如果要改权重定义、默认 cut 或 target/background 归类，请先确认这会不会破坏旧结果的可比较性。

## 对后续代理最有用的工作习惯

- 先读 `README.md` 和目标 config，再决定是否需要改 Python。
- 用户若只是想“换样本、换默认 target、换输出目录、换 plot preset”，大概率只需要改 JSON。
- 用户若只是想调 event-BDT 的样本、feature、preselection、reweighting 变量，也优先改 `config/event_bdt/*.json`。
- 用户若只是想换 NN input、模型 score 分支、truth/sample composition 或 selection，优先改 `config/nn_study/*.json`。
- 只改绘图时，优先走 `--plot-only`，不要每次都重读 ROOT。
- 只改 merge/Condor 逻辑时，不要碰 physics 分类代码。
- 只改 physics 定义时，要留意 histogram payload 和 text/report 输出是否同步。

## 验证建议

这个仓库当前没有看到独立测试套件，因此改动后至少做下面其中一项：

- 语法检查：

```bash
python3 -m compileall src/tthcc_an src/tthcc_an/event_bdt scripts
```

- 小范围本地运行：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --targets hcc \
  --scores gpart_h2cc \
  --max-files-per-sample 1 \
  --outdir outputs/boosted_higgs_tagger_study_2024_smoke
```

- 若要验证 `xbb-vs-xcc` 的 `tight` contour / label / CMS style：

```bash
python scripts/run_boosted_higgs_tagger_study.py \
  --config config/samples_2024_add_nonttbarmatch_allmc.json \
  --plot-only \
  --plot-input outputs/boosted_higgs_tagger_study_2024_tight_20260525/plot_input.npz \
  --outdir outputs/boosted_higgs_tagger_study_2024_tight_20260525 \
  --xbb-vs-xcc-region-preset tight
```

- event-BDT 原型验证：

```bash
python scripts/run_event_bdt.py \
  evaluate \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json

python scripts/run_event_bdt.py \
  evaluate \
  --config config/event_bdt/train_ttHcc_0l_4class_baseline_jecs.json
```

- 若刚改了 event-BDT 的 analysis branches：

```bash
python scripts/run_event_bdt.py \
  prepare \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json \
  --force

python scripts/run_event_bdt.py \
  evaluate \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json
```

- event-BDT 打分示例：

```bash
python scripts/run_event_bdt.py \
  predict \
  --config config/event_bdt/train_ttHcc_0l_3class_baseline_jecs.json \
  --outdir outputs/event_bdt_ttHcc_0l_3class_baseline_jecs_v3/scored_root
```

- event-level NN study 验证：

```bash
python scripts/run_nn_study.py \
  --config config/nn_study/nn_study_2024_1l_v1.json \
  --max-files-per-sample 1 \
  --outdir /tmp/nn_study_2024_1l_v1_smoke
```

至少检查 summary 中：

- `number_of_events_unclassified == 0`
- confusion normalization 没有 NaN
- pairwise AUC 全在 `[0, 1]`
- `auc_validation.max_abs_difference` 与 sklearn 一致到浮点精度
- sample selection 没有把 inclusive `TTto*` bottom 与专用 bottom samples 重复计数

如果改动涉及 payload schema，至少验证：

- `--export-chunk`
- `--merge-chunks`
- `--plot-only`

这三条链路是否仍兼容。

## 不要轻易动的内容

- `outputs/` 和 `condor/` 里的生成物，除非用户明确要求清理或重建
- truth code / process code 的既有编码顺序
- 已经在 README 和 config 中公开使用的 CLI 选项名
- chunk payload metadata key，除非愿意一起做兼容迁移

## 常见落点

- “为什么配置里能写注释？”  
  因为 JSON 实际是通过 `strip_hash_comments()` 预处理后再解析。

- “为什么命令行没传某个参数也生效了？”  
  因为 config 会回填默认值，且优先级是 CLI > config > fallback。

- “为什么改图不需要重跑所有 ROOT？”  
  因为仓库会缓存 `plot_input.npz`，并支持 `--plot-only`。

- “为什么我更新了 ntuple / `gen_sumws.json` 但结果没变？”  
  因为你大概率只跑了 `--plot-only`；这种情况下必须重新读 ROOT 或重新走 chunk merge。

- “为什么 Condor 默认是 chunked？”  
  因为推荐流程是 many chunk jobs 导出 slim NPZ，再由 merge job 汇总，内存压力更低。

- “为什么 event-BDT 没有直接在文件顶部 `import xgboost as xgb`？”  
  因为当前实现把 `xgboost` 延迟到训练时才导入，这样 `prepare` 和 `--help` 在没有 ML 依赖时也还能工作。

## 交付预期

在这个仓库里，好的代理改动通常应满足：

- 配置驱动优先
- 输出目录和文件命名保持稳定
- physics 定义改动有文档同步
- 绘图、payload、reporting 不出现半同步状态
- 若没有完整跑大样本，明确说明验证范围
