# GitHub 上传说明

## 1. 解压与本地运行
解压 `puffy-scattering-ai-github.zip`，进入含 README.md 的仓库目录。
需要 Python 3.11–3.13；本机验证使用 3.12。首次安装需要能访问 Python 包索引。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
python examples/quickstart.py
python examples/train_from_csv.py
python -m puffy benchmark --out results/my-run --ntrain 24 --ntest 20 --active 6 --seed 2026
```
Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`。
若系统的 `python` 不存在，请使用 `python3`（Windows 可用 `py`）。
最后一条会重新求解、训练、主动选点并生成 CSV 与指标 JSON；耗时取决于机器。
图为可选：`python -m pip install -e '.[plot]'`，然后 `python -m puffy.plot_results results/my-run`。

## 2. 创建空 GitHub 仓库
登录 GitHub，创建名为 `puffy-scattering-ai` 的仓库。不要勾选自动创建 README、LICENSE 或 .gitignore，因为包内已包含这些文件。

## 3. 上传整个解压目录
在解压后的仓库根目录运行（将 YOUR_USERNAME 改为自己的用户名）：

```bash
git init
git add .
git status
git commit -m "Add independent finite-size scattering solver and AI benchmark"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/puffy-scattering-ai.git
git push -u origin main
```
首次推送按 GitHub 提示完成身份验证，不要将密码或令牌写入文件。
`git status` 应能看到 `.github/workflows/tests.yml` 和 `.gitignore`；这两个隐藏路径必须上传。
不要把 zip 本身作为唯一文件上传。Git 命令能保留隐藏文件，比手动拖拽更可靠。

## 4. 查看 CI
推送后打开仓库 Actions 页查看 Tests。工作流配置了 Python 3.11、3.12、3.13，执行安装、测试、示例、notebook 与构建。
本次只验证本地环境，尚未在 GitHub 服务器执行 CI，也没有自动替你创建远程仓库。

## 5. 来源与发布范围
论文/Mathematica 附件不打包。原文件可放在被忽略的 references/ 中供本地静态检查。
MIT 范围见 LICENSE 与 THIRD_PARTY_NOTICES.md；不要将其解释为对原论文/notebook 的再授权。
公开数据为独立 Python 求解器生成；论文原始数据、Wolfram 运行结果和未实现的神经网络均不在此仓库。
