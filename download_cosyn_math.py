import os
from huggingface_hub import hf_hub_download

def download_cosyn_math(save_dir: str = "./CoSyn_400k_math"):
    """
    下载 HuggingFaceM4/FineVision 数据集中的 CoSyn_400k_math 所有分片
    并保存到本地目录
    """
    repo_id = "HuggingFaceM4/FineVision"
    subdir = "CoSyn_400k_math"

    # 数据集一共 13 个 parquet 文件
    filenames = [f"train-{i:05d}-of-00013.parquet" for i in range(13)]

    os.makedirs(save_dir, exist_ok=True)

    for fname in filenames:
        print(f"📥 正在下载 {fname} ...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/{fname}",
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ 已保存到: {local_path}")

    print("\n🎉 全部下载完成！数据保存在:", os.path.abspath(save_dir))


if __name__ == "__main__":
    download_cosyn_math()
