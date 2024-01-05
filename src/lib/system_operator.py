import subprocess
import re


def get_gpu_memory_usage():
    # nvidia-smi コマンドを実行
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
    # 出力をデコード
    result = result.decode('utf-8')

    # GPUメモリの使用量と総メモリを抽出
    gpu_memory = [list(map(int, re.findall(r'\d+', line)))
                  for line in result.strip().split('\n')]

    # 使用率を計算
    usage = [(used / total) * 100 for used, total in gpu_memory]
    return str(usage[0])[:5]
