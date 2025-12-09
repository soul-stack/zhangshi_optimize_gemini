# -*- coding: utf-8 -*-
"""
渗氮时间反推服务（Sanic 版，字段与 MES 对齐）
- 目标中值 & 最短时间：优先从 GUI 生成的 product_config.json 获取

MES → Python（POST /calc_time）:
{
    "lineCode": "A",
    "emtType": "B",
    "guaWeight": "520.85",
    "rclTid": "998",
    "minTime": "15"  // 单炉反应最短时间（字符串）
}

Python → MES (成功):
{
    "status": "ok",
    "data": {
        "lineCode": "A",
        "emtType": "B",
        "guaWeight": "520.85",
        "rclTid": "998",
        "danTime": "100.00"
    }
}

Python → MES (失败):
{
    "status": "error",
    "message": "错误描述..."
}
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from sanic import Sanic
from sanic.response import json as sanic_json
import sklearn  # noqa: F401

os.environ.setdefault("SANIC_TOUCHUP", "false")


# ==================== 公共数据根目录 ====================
def get_data_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir.parent
    return Path(__file__).resolve().parent


DATA_ROOT = get_data_root()
ART_DIR = DATA_ROOT / "bp_artifacts"

MODEL_PATH = ART_DIR / "best_bp_model_final.pth"
SCALER_PATH = ART_DIR / "scalers.pkl"

# 配置文件
FURNACE_PARAM_FILE = DATA_ROOT / "furnace_params.json"
PRODUCT_CONF_FILE = DATA_ROOT / "product_config.json"  # 产品配置

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 公共工具 ====================
def safe_exit_with_error(msg: str):
    print("\n[致命错误] " + msg)
    print("\n详细信息：")
    traceback.print_exc()
    if getattr(sys, "frozen", False):
        input("\n按回车键退出...")
    sys.exit(1)


# ==================== 加载 BP 工件 ====================
import pickle

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到 BP 模型文件：{MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"未找到归一化器文件：{SCALER_PATH}")

    with open(SCALER_PATH, "rb") as f:
        scaler_obj = pickle.load(f)

    x_scaler = scaler_obj["x_scaler"]
    y_scaler = scaler_obj["y_scaler"]
    feature_order = scaler_obj.get("feature_order", ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度"])

    config_path = ART_DIR / "best_config_final.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("arch", [64, 32])
    else:
        arch = [64, 32]


    class BPNet(nn.Module):
        def __init__(self, in_dim: int, arch_list, out_dim: int = 1):
            super().__init__()
            layers = []
            last = in_dim
            for h in arch_list:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ReLU())
                last = h
            layers.append(nn.Linear(last, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    bp_model = BPNet(in_dim=len(feature_order), arch_list=arch, out_dim=1).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    bp_model.load_state_dict(state_dict)
    bp_model.eval()

except Exception:
    safe_exit_with_error("加载 BP 模型/配置 失败，请检查 bp_artifacts 目录和相关文件是否存在且完整。")


# ==================== 配置读取 (产品 & 炉子) ====================
def load_furnace_params() -> Dict[str, Any]:
    if not FURNACE_PARAM_FILE.exists():
        return {}
    try:
        with open(FURNACE_PARAM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_product_config() -> Dict[str, Dict[str, Any]]:
    """读取产品配置文件，获取目标值和最短时间"""
    if not PRODUCT_CONF_FILE.exists():
        # 若文件不存在（例如只启动了 API 还没开过 GUI），返回空或硬编码兜底
        return {}
    try:
        with open(PRODUCT_CONF_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_line_params(line: str) -> Tuple[float, float, float, float]:
    params_all = load_furnace_params()

    def _extract(line_name: str) -> Optional[Tuple[float, float, float, float]]:
        info = params_all.get(line_name)
        if not info: return None
        furnaces = info.get("furnaces", {})
        temps, airs, salts = [], [], []
        for fname in ("N2", "N3", "N4", "N5"):
            fi = furnaces.get(fname, {}) or {}
            if not bool(fi.get("enabled", True)):
                continue
            if fi.get("temperature") is not None: temps.append(float(fi["temperature"]))
            if fi.get("air_flow") is not None: airs.append(float(fi["air_flow"]))
            if fi.get("salt") is not None: salts.append(float(fi["salt"]))
        if not temps or not airs or not salts:
            return None
        return float(np.mean(temps)), float(np.mean(airs)), float(np.mean(salts)), float(info.get("time_adjust", 0.0))

    res = _extract(line)
    if res is not None: return res
    res = _extract("A")
    if res is not None: return res
    return 585.0, 8.0, 3.0, 0.0


def get_target_info(product_type: str) -> Tuple[float, float]:
    """
    根据产品类型获取 (目标厚度, 最短时间)
    逻辑：读取产品配置文件 -> 匹配 -> 默认兜底
    """
    conf = load_product_config()
    pt = product_type.upper()

    # 默认兜底值 (A)
    default_target = 9.5
    default_min_time = 48.0

    if pt in conf:
        info = conf[pt]
        t = float(info.get("target", default_target))
        m = float(info.get("min_time", default_min_time))
        return t, m

    # 尝试找第一个
    if len(conf) > 0:
        first_key = list(conf.keys())[0]
        info = conf[first_key]
        return float(info.get("target", default_target)), float(info.get("min_time", default_min_time))

    return default_target, default_min_time


# ==================== 预测 & 寻优 ====================
def bp_predict_thickness(weight: float, hold_time: float, salt: float, air_flow: float, temp: float) -> float:
    feat = np.zeros((1, len(feature_order)), dtype=np.float32)
    values = {"挂重量": weight, "实际时长": hold_time, "平均加盐量": salt, "通气量": air_flow, "平均温度": temp}
    for i, name in enumerate(feature_order):
        feat[0, i] = values.get(name, 0.0)
    feat_n = x_scaler.transform(feat)
    with torch.no_grad():
        y_n = bp_model(torch.tensor(feat_n, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return float(y_scaler.inverse_transform(y_n).ravel()[0])


def find_best_time(weight: float, target_mid: float, avg_salt: float, avg_air: float, avg_temp: float, t_min: int = 40,
                   t_max: int = 140) -> Tuple[int, float]:
    best_t = t_min
    best_pred = None
    best_err = float("inf")
    for t in range(t_min, t_max + 1):
        pred = bp_predict_thickness(weight, t, avg_salt, avg_air, avg_temp)
        err = abs(pred - target_mid)
        if err < best_err:
            best_err = err
            best_t = t
            best_pred = pred
    if best_pred is None:
        best_pred = bp_predict_thickness(weight, best_t, avg_salt, avg_air, avg_temp)
    return best_t, best_pred


# ==================== Sanic 应用 ====================
app = Sanic("NitridingTimeService")


@app.post("/calc_time")
async def calc_time(request):
    try:
        payload = request.json or {}
    except Exception:
        return sanic_json({"status": "error", "message": "请求体不是合法 JSON"}, status=400)

    line_code = str(payload.get("lineCode", "A")).strip().upper()
    if line_code not in ("A", "B", "C"): line_code = "A"

    emt_type = str(payload.get("emtType", "")).strip()
    rcl_tid = str(payload.get("rclTid", "")).strip()
    gua_weight_raw = payload.get("guaWeight", "0")

    # 1. 解析 minTime (单炉最短时间)
    req_min_time_single = 0.0
    try:
        req_min_time_str = str(payload.get("minTime", "0"))
        req_min_time_single = float(req_min_time_str)
    except:
        req_min_time_single = 0.0

    if not emt_type or not rcl_tid:
        return sanic_json({"status": "error", "message": "缺少 emtType 或 rclTid"}, status=400)

    try:
        gua_weight = float(str(gua_weight_raw))
    except:
        return sanic_json({"status": "error", "message": "guaWeight 必须为数字"}, status=400)

    # 2. 获取炉子环境参数 (avg_temp, avg_air, avg_salt, time_adjust)
    avg_temp, avg_air, avg_salt, time_adjust = get_line_params(line_code)

    # 3. 获取产品目标参数 (从 GUI 配置获取)
    #    target_val: 优化目标厚度
    #    config_min_time: GUI 配置表中的最短时间 (通常是总时间)
    target_val, config_min_time = get_target_info(emt_type)

    # 4. 确定最终的“总时长下限”
    #    逻辑：取 [GUI配置的最短时间] 与 [MES传入单炉时间 * 4] 中的较大值
    limit_from_mes_total = req_min_time_single * 4.0
    final_floor_time = max(config_min_time, limit_from_mes_total)

    # 5. 寻优计算 (基准时间)
    base_time, base_pred = find_best_time(
        weight=gua_weight,
        target_mid=target_val,
        avg_salt=avg_salt,
        avg_air=avg_air,
        avg_temp=avg_temp
    )

    # 6. 最终计算
    #    总时间 = max( (模型基准时间 + 产线微调), 总时长下限 )
    calculated_total = base_time + time_adjust
    total_time = max(calculated_total, final_floor_time)

    # 7. 构造返回结果 (包装 status 和 data)
    result_data = {
        "lineCode": line_code,
        "emtType": emt_type,
        "guaWeight": f"{gua_weight:.2f}",
        "rclTid": rcl_tid,
        "danTime": f"{total_time:.2f}"
    }

    return sanic_json({"status": "ok", "data": result_data}, ensure_ascii=False)


if __name__ == "__main__":
    try:
        print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
        print(f"[INFO] 产品配置路径 = {PRODUCT_CONF_FILE}")
        app.run(host="0.0.0.0", port=8000, single_process=True, access_log=True)
    except Exception:
        safe_exit_with_error("Sanic 服务启动失败。")