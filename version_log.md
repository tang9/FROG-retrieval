# Version Log

## Unreleased — 2026-08-15

### Unified GUI

- 在 `Save All` 左侧新增 `Convert + Retrieval`，依次执行数据转换和脉冲反演。
- 在 `Save All` 右侧新增 `Run All`，依次执行 Convert、Retrieval、Result 数据传递和 Save All。
- 串联流程会等待异步 Retrieval 完成后再继续，运行期间禁用重复启动按钮，并在失败后恢复按钮状态、显示错误信息。
- `Save All` 适配内嵌的 Phase Analysis 页面。

### Convert 界面

- 将 `Load Param` 和 `Save Param` 移到 `Save Default Param` 上方。
- 将 `time_min`、`time_max` 合并到 `delay range` 同一行，并移除重复标签。
- 调整 Parameters 面板、标签和输入区域的比例与间距，使输入框布局更紧凑。
- 统一底部操作按钮宽度；窗口拉伸时按钮保持固定宽度，额外空间位于 `Trace log scale` 和 `Show 1D` 之间。
- 使用独立的 `LP delay scale (fs)` 和 `LP wavelength scale (nm)` 代替界面中的 `cutoff ratio`，支持分别控制 delay 和 wavelength 方向的低通尺度。
- 加载旧参数文件时，自动把 `cutoff`、`cutoff_ratio` 或 `cutoff ratio` 换算为新的 delay/wavelength low-pass scale；尚未加载数据时延迟换算，保留向后兼容。

### Retrieval

- 新增可选的 `delay_smearing (fs)` 参数；只有勾选后才应用沿 delay 轴的高斯仪器响应展宽。
- Python 和 Cython 两条计算路径均支持 delay smearing，并在误差计算及最终 reconstructed trace 中保持一致。
- 优化 Python 路径的误差计算、数组处理和多项式项累积，减少重复计算与临时数组。
- 对 Cython 路径进行了对应优化，包括复用误差上下文、合并 G/G′ 计算以及在编译代码中执行 delay-axis convolution。

### Result 与 Phase Analysis

- 将 Phase Analysis 放入 FROG Result 界面的第二个标签页，移除原独立按钮和弹窗。
- Phase Analysis 绘图改用 Matplotlib，并支持直接保存分析图。
- Parameters 中 `time_min`、`time_max` 改变后，会同步刷新 Phase Analysis 的时间范围。
- 修正左图双纵坐标标签重叠问题。
- 左图频谱强度和右图时域 intensity profile 的纵坐标范围统一为 `0–1.4`。

### 输出路径

- 统一 Convert、Retrieval 和 Result 的保存路径生成逻辑。
- 已位于 `retrieval_result` 目录中的结果会复用现有目录，避免反复生成嵌套的 `retrieval_result/retrieval_result` 文件夹。

### 兼容性与验证

- 旧版 low-pass `cutoff ratio` 参数文件仍可加载并自动迁移。
- Cython 生成的 C 文件已同步更新。
- 自动化测试覆盖 low-pass 参数迁移、delay smearing、Python/Cython 数值一致性、输出路径、Phase Analysis 标签页及统一运行流程。
- 当前测试结果：22 项全部通过。
