## 这个 PR 做了什么

<!-- 一两句话说明改动与动机 -->

## 改动范围

- [ ] ASR / 说话人后端
- [ ] 渲染与输出结构
- [ ] 缓存、CLI、性能
- [ ] 文档
- [ ] 校订口径（`src/processor/rules.py`）——若是，请先确认对应 issue 已讨论

## 验证

- [ ] `pytest -q` 全量通过
- [ ] `ruff check .` 无新增告警
- [ ] 若改动影响产物结构：`python -m src.processor.session_edit --check <本地成稿>` 通过
- [ ] 若改动影响转录或校订效果：附上**实际跑过的节目与逐字对照**（改前 / 改后）

<!-- 效果类改动请贴出对照片段，只写「更准了」无法验证 -->

## 需要确认的点

- [ ] 是否触碰 `src/processor/rules.py`（两条链路共享的口径源）？若是，`RULES_SPEC_VERSION` 已递增：
- [ ] 是否同时更新了校验器（`validate_session_output`）与渲染器（`src/renderer/markdown.py`）？
- [ ] 是否引入了新的第三方依赖？若是，请说明为何不能省：

## 其他

<!-- 有需要讨论的取舍写在这里 -->
