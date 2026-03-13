# parse_object_json 设计说明

## 目标

从 `dataset_index_all.json` 与各样本的 `object.json` 解析 `diffuse_tree`，生成标准化的 `object_meta.json`，为后续「逻辑部件级 mask 渲染」与「joint-aware bbox token 监督」提供统一、稳定、可解释的数据格式。

## 输入

- **dataset_index_all.json**：由 `scan_dataset.py` 生成，每行一条样本，含 `category`, `model_id`, `object_dir`, `object_json`, `objs_dir`, `plys_dir` 等。
- **object.json**：每个物体一份，内含 `diffuse_tree`（或整文件即树根），表示 articulated 部件树。

## 输出

- **object_meta.json**（每个物体一份）：路径为  
  `../project/processed_data/objects_meta/<category>/<model_id>/object_meta.json`
- **object_meta_index.json**：总索引，记录每份 `object_meta.json` 的路径。
- **logs/parse_object_json.log**：整体运行日志。
- **stats/object_meta_stats.json**：统计摘要（成功/失败数、每类样本数、每类平均 part 数、各 joint type 计数）。
- **debug_samples/**：随机 3～5 个物体的解析中间结果，便于人工检查。

## object_meta.json 字段设计

| 层级 | 字段 | 含义 | 来源/说明 |
|------|------|------|-----------|
| 顶层 | object_id | 唯一标识 | `{category}_{model_id}` |
| | category / model_id | 类别与模型 ID | 与 dataset index 一致 |
| | source_object_dir / source_object_json | 原始路径 | 可追溯回原始数据 |
| | meta | 物体级统计与摘要 | 见下 |
| | part_order | 部件顺序 | 第一版：按 part_id 升序 |
| | parts | 部件列表 | 每项对应一个逻辑 part |
| meta | obj_cat, depth, n_arti_parts, n_revolute, n_prismatic, n_diff_parts, tree_hash | 统计与可复现性 | 从树与 joint 汇总 |
| parts[] | part_id, parent_id, children, name | 树结构 | 从 diffuse_tree 遍历得到 |
| | joint (type, range, axis_origin, axis_direction) | 关节参数 | 缺省时用固定值并打日志 |
| | bbox (center, size) | 包围盒 | 有则用，无则默认并打日志 |
| | mesh_files, ply_files | 逻辑 part 对应的文件列表 | 一个 part 可对应多 obj/ply |
| | is_articulated, is_leaf | 是否可动、是否叶子 | 由 joint 与 children 推导 |

## diffuse_tree 解析约定

- **逻辑 part**：树中每个 node 对应一个逻辑 part；一个 node 可包含多个 `objs`/`plys`，渲染与 joint 均按逻辑 part 为单位。
- **树遍历**：使用 BFS 为每个 node 分配 `part_id`（0, 1, 2, …），保证顺序稳定；`part_order` 取为 `sorted(part_id)`，即 `[0, 1, …, N-1]`。
- **鲁棒性**：node 无 `joint` 视为 `fixed`；无 `axis_origin`/`axis_direction` 用 `[0,0,0]` 并记录；无 `bbox` 用 center=[0,0,0]、size=[0,0,0] 并记录；`objs`/`plys` 为空则列表为空。

## part_order 为何先按 part_id 升序

- **第一版**：`part_order = sorted(all part_ids)`，即按 part_id 升序。优点：实现简单、与 BFS 分配的 part_id 一致、便于调试和复现；后续 token 顺序与 part_id 一一对应。
- **若改为 BFS/DFS 顺序**：只需修改「生成 part_order 」的步骤（例如改为按 BFS 序或 DFS 序的 part_id 列表），并在 README 中说明顺序定义；解析与 part 内容不变。

## 后续使用 object_meta.json 的模块

- **part mask 渲染**：按 `parts[].mesh_files` 合并 mesh，按 `part_id` 渲染 part-level segmentation。
- **joint token 生成**：从 `parts[].joint` 与 `bbox`、`parent_id` 等生成 joint-aware bbox token 监督。
- **bboxgen 微调**：用 part_order 与 bbox/joint 构建训练用的 token 序列与标签。
