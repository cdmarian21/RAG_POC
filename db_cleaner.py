import shutil
import os
import config

if os.path.exists(config.DB_PATH):
    shutil.rmtree(config.DB_PATH)
    print(f"deleted db: {config.DB_PATH}")
else:
    print("DB empty or doesn't exist")