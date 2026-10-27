"""保留旧启动命令；实现已拆分到 main.py 和 harness/。"""

from main import main


if __name__ == "__main__":
    raise SystemExit(main())
