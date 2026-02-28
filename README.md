# 草本智营（课程设计版）

基于“知识图谱 + RAG + 多智能体”思想实现的中医企业智能营销系统 MVP。

## 功能概览

- 产品知识上传（结构化存储）
- 一键生成小红书种草文案
- 智能客服问答（RAG 检索 + 图谱提示）
- 合规检测（高风险词规则）

## 技术栈

- Python 3.10+
- Streamlit（界面）
- scikit-learn（TF-IDF 检索）
- networkx（轻量知识图谱）
- pandas / numpy

## 项目结构

- `t.py`：主程序（Streamlit 应用）
- `requirements.txt`：依赖列表

## 使用 conda 创建虚拟环境

在 PowerShell 中执行：

```bash
conda create -n caoben_zhiying python=3.10 -y
conda activate caoben_zhiying
pip install -r requirements.txt
```

## 启动系统

```bash
streamlit run t.py
```

启动后浏览器访问本地地址（通常为 `http://localhost:8501`）。

## 模块说明

### 1) 产品知识上传
输入产品名称、功效、成分、目标人群，系统自动加入知识库与图谱。

### 2) 小红书内容生成
输入主题（产品 + 人群），系统结合语义检索结果生成种草文案。

### 3) 智能问答客服
输入用户问题，系统返回科普型问答结果并给出图谱关系提示。

### 4) 合规检测
自动检测文案中的高风险医疗宣称词（如“治疗”“根治”“保证”等）。

## 推送到 GitHub（你的仓库）

你给出的仓库地址：`https://github.com/Dongmain/LG.git`

在项目目录执行：

```bash
git init
git add .
git commit -m "feat: build herb intelligent marketing MVP with Streamlit"
git branch -M main
git remote add origin https://github.com/Dongmain/LG.git
git push -u origin main
```

如果推送时需要登录：
- 建议使用 GitHub Personal Access Token（PAT）作为密码
- 或提前执行 `gh auth login` 完成认证

## 说明

本项目用于课程设计演示，重点在“系统架构与方法验证”，非医疗建议，不替代专业医生诊疗。