#!/usr/bin/env bash
# 在国内 Ubuntu 服务器上安装 GitHub Actions self-hosted runner，用于跑 00-daily-analysis.yml。
#
# 用法（以 root 或有 sudo 的用户执行）：
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/setup_cn_runner.sh \
#     | sudo bash -s -- <owner>/<repo> <registration-token>
#
# registration-token 由仓库管理员生成（1 小时内有效）：
#   gh api -X POST repos/<owner>/<repo>/actions/runners/registration-token --jq .token
# 或网页：Settings → Actions → Runners → New self-hosted runner。
#
# 做了什么：
#   1. 安装 python3 / venv / 构建依赖；pip 走清华镜像
#   2. 创建专用用户 runner，下载最新 actions-runner（从 GitHub，国内可能慢，会重试）
#   3. 以 label "self-hosted,cn" 注册，并安装为 systemd 服务，开机自启
# 之后在仓库 Settings → Variables 设 RUNNER_LABEL=self-hosted 即生效。
set -euo pipefail

REPO="${1:-}"; TOKEN="${2:-}"
if [ -z "$REPO" ] || [ -z "$TOKEN" ]; then
  echo "用法: $0 <owner>/<repo> <registration-token>" >&2; exit 1
fi
if [ "$(id -u)" -ne 0 ]; then echo "请用 root 或 sudo 执行" >&2; exit 1; fi

RUNNER_USER=runner
RUNNER_HOME=/opt/actions-runner

echo "==> 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl tar jq git python3 python3-venv python3-pip build-essential libffi-dev libssl-dev ca-certificates >/dev/null

echo "==> 配置 pip 国内镜像（全局）"
mkdir -p /etc/pip
cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 60
EOF

echo "==> 创建 runner 用户"
id -u "$RUNNER_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$RUNNER_USER"
mkdir -p "$RUNNER_HOME"; chown "$RUNNER_USER:$RUNNER_USER" "$RUNNER_HOME"

echo "==> 下载 actions-runner"
ARCH=$(uname -m); case "$ARCH" in x86_64) RARCH=x64;; aarch64) RARCH=arm64;; *) echo "不支持的架构 $ARCH" >&2; exit 1;; esac
VER=$(curl -fsSL --retry 5 https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name | sed 's/^v//')
TARBALL="actions-runner-linux-${RARCH}-${VER}.tar.gz"
cd "$RUNNER_HOME"
if [ ! -f ".runner" ]; then
  sudo -u "$RUNNER_USER" curl -fL --retry 5 --retry-delay 5 -o "$TARBALL" "https://github.com/actions/runner/releases/download/v${VER}/${TARBALL}"
  sudo -u "$RUNNER_USER" tar xzf "$TARBALL" && rm -f "$TARBALL"
  ./bin/installdependencies.sh >/dev/null
  echo "==> 注册 runner（label: self-hosted,cn）"
  sudo -u "$RUNNER_USER" ./config.sh --unattended \
    --url "https://github.com/${REPO}" --token "$TOKEN" \
    --name "cn-$(hostname)" --labels "cn" --work _work --replace
else
  echo "已存在注册信息，跳过 config.sh"
fi

echo "==> 安装 systemd 服务并启动"
./svc.sh install "$RUNNER_USER" >/dev/null
./svc.sh start >/dev/null
sleep 2; ./svc.sh status | head -5

echo
echo "完成。到 https://github.com/${REPO}/settings/actions/runners 应能看到 Idle 状态。"
echo "然后在 Settings → Variables 添加 RUNNER_LABEL=self-hosted 即切换；删除该变量回退到 GitHub 托管机。"
