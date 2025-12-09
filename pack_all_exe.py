# pack_all_exe.py
# 一键打包：BP 训练 + 炉子维护 GUI + 接口服务（中文 exe 名称）
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent
# 建议在已安装依赖的虚拟环境中运行此脚本
PYTHON_EXE = sys.executable


def run(cmd: list[str]):
    print("\n▶️ 运行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)


def check_file(name: str) -> Path | None:
    """检查指定脚本是否存在，存在则返回路径，不存在则打印提示并返回 None"""
    p = BASE_DIR / name
    if not p.exists():
        print(f"❌ 找不到 {name}，跳过打包该脚本")
        return None
    return p


def add_data_arg(args: list[str], rel_path: str, dest: str):
    """只有当文件存在时才追加 --add-data 参数"""
    src = BASE_DIR / rel_path
    if src.exists():
        args += ["--add-data", f"{src}{os.pathsep}{dest}"]
    else:
        print(f"⚠️ 跳过缺失数据文件：{src}")


def copy_shared_files_to_dist():
    """把共享配置/模型复制到 dist 根目录，供三个 exe 共同使用"""
    dist_root = BASE_DIR / "dist"
    dist_root.mkdir(exist_ok=True)

    shared_items = [
        "bp_artifacts",
        "furnace_params.json",
        "furnace_log_config.json",
        "product_config.json",      # ✅ 确保产品配置被复制
        "furnace_time_history.csv",
        "line_time_adjust.json",
        "time_adjust.json",
        "furnace_adjust_log.xlsx",
        "furnace_adjust_log1.xlsx",
    ]

    for item in shared_items:
        src = BASE_DIR / item
        dst = dist_root / item
        if not src.exists():
            print(f"⚠️ 共享文件不存在，略过：{src}")
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"✅ 已复制共享文件到 dist：{dst}")


def main():
    os.chdir(BASE_DIR)
    print("当前目录：", BASE_DIR)
    print("使用解释器：", PYTHON_EXE)

    # PyInstaller 公共参数
    common_args = ["-y", "--clean", "--log-level=WARN"]

    # ========== 1) BP 训练脚本 → 训练.exe ==========
    bp_script = check_file("bp_train_with_specs.py")
    if bp_script is not None:
        try:
            args = [
                PYTHON_EXE, "-m", "PyInstaller",
                *common_args,
                "--name", "训练",            # 中文 exe 名称
                str(bp_script)
            ]
            run(args)
            print("✅ BP 训练 exe：dist/训练/训练.exe")
        except subprocess.CalledProcessError:
            print("⚠️ BP 训练脚本打包失败（不影响后续）")

    # ========== 2) 炉子维护 GUI → 热处理维护.exe ==========
    gui_script = check_file("furnace_time_manager_gui.py")
    if gui_script is not None:
        try:
            args = [
                PYTHON_EXE, "-m", "PyInstaller",
                *common_args,
                "--name", "热处理维护",      # 中文 exe 名称
                "--windowed",              # GUI 程序无控制台
                "--collect-all", "sklearn",
                str(gui_script)
            ]
            # 仅在存在时添加数据文件
            add_data_arg(args, "bp_artifacts", "bp_artifacts")
            add_data_arg(args, "furnace_params.json", ".")
            add_data_arg(args, "furnace_log_config.json", ".")
            add_data_arg(args, "furnace_adjust_log.xlsx", ".")
            add_data_arg(args, "furnace_adjust_log1.xlsx", ".")
            add_data_arg(args, "furnace_time_history.csv", ".")
            add_data_arg(args, "line_time_adjust.json", ".")
            add_data_arg(args, "time_adjust.json", ".")

            run(args)
            print("✅ 炉子维护 GUI exe：dist/热处理维护/热处理维护.exe")
        except subprocess.CalledProcessError:
            print("⚠️ 炉子维护 GUI 打包失败")

    # ========== 3) 接口服务 → 接口链接.exe （无控制台后台程序） ==========
    service_script = check_file("nitriding_service_zh_api.py")
    if service_script is not None:
        try:
            args = [
                PYTHON_EXE, "-m", "PyInstaller",
                *common_args,
                "--name", "接口链接",       # 中文 exe 名称
                "--noconsole",             # 关键：隐藏控制台窗口，后台运行
                "--collect-all", "sanic",
                "--collect-all", "tracerite",
                "--copy-metadata", "html5tagger",
                "--collect-all", "sklearn",
                str(service_script)
            ]
            add_data_arg(args, "bp_artifacts", "bp_artifacts")
            add_data_arg(args, "furnace_params.json", ".")
            run(args)
            print("✅ 接口服务 exe：dist/接口链接/接口链接.exe")
        except subprocess.CalledProcessError:
            print("⚠️ 接口服务打包失败")

    # ========== 4) 复制共享数据到 dist 根目录 ==========
    copy_shared_files_to_dist()

    print("\n🎉 全部打包流程结束，请从 dist/ 目录使用 exe 与共享配置。")
    print("   - 训练：dist/训练/训练.exe")
    print("   - 热处理维护：dist/热处理维护/热处理维护.exe")
    print("   - 接口链接：dist/接口链接/接口链接.exe（无控制台后台程序）")


if __name__ == "__main__":
    main()