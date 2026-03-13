# object_meta.json 与 parse_object_json 说明

## 一、脚本解决什么问题

parse_object_json.py 读取 dataset_index_all.json 与各物体的 object.json，解析 diffuse_tree，生成标准化的 object_meta.json，为「逻辑部件级 mask 渲染」与「joint-aware bbox token 监督」提供统一、稳定、可解释的数据格式。

## 二、输入

- dataset_index_all.json（默认 ../project/processed_data/dataset_index_all.json）
- 每个物体一份 object.json（路径来自 index 的 object_json 或 object_dir/object.json）

## 三、输出

- object_meta.json：../project/processed_data/objects_meta/<category>/<model_id>/object_meta.json
- object_meta_index.json：../project/processed_data/indexes/object_meta_index.json
- parse_object_json.log：../project/processed_data/logs/parse_object_json.log
- object_meta_stats.json：../project/processed_data/stats/object_meta_stats.json
- debug_samples/：随机 3～5 个物体的完整 object_meta，便于人工检查

## 四、object_meta.json 各字段含义

- 顶层：object_id, category, model_id, source_object_dir, source_object_json, meta, part_order, parts
- meta：obj_cat, depth, n_arti_parts, n_revolute, n_prismatic, n_diff_parts, tree_hash
- parts[] 每项：part_id, parent_id, children, name, joint(type, range, axis_origin, axis_direction), bbox(center, size), mesh_files, ply_files, is_articulated, is_leaf

## 五、为什么把一个 diffuse_tree node 视为逻辑 part

一个 node 可能对应多个 obj/ply 文件；按单文件渲染会拆开同一逻辑部件。约定一个 node = 一个逻辑 part，渲染时需合并该 part 的 mesh_files 再按 part_id 渲染。

## 六、为什么 part_order 先按 part_id 升序

第一版 part_order = sorted(part_id)。part_id 已按 BFS 分配为 0,1,2,...，升序即 [0,1,...,N-1]，实现简单、确定、易复现。若改为 BFS/DFS 顺序，只需在 parse_object_json.py 的 build_part_order 中改为按遍历顺序收集 part_id，并在此 README 说明新顺序即可。

## 七、后续哪些模块会使用 object_meta.json

- part mask 渲染：按 parts[].mesh_files 合并 mesh，按 part_id 渲染 part-level segmentation
- joint token 生成：从 parts[].joint、bbox、parent_id 等生成 joint-aware bbox token 监督
- bboxgen 微调：用 part_order 与 bbox/joint 构建训练 token 与标签