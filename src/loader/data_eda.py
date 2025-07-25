# EDA

import os

pdf_files = []
hwp_files = []

# 파일 절대 경로
data_dir = "/home/daeseok/AI-Engineer/data"

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        pdf_files.append(filename)
    elif filename.endswith(".hwp"):
        hwp_files.append(filename)    
        
print(f"총 파일 수: {len(os.listdir(data_dir))}")
print(f"PDF 파일 수: {len(pdf_files)}")
print(f"HWP 파일 수: {len(hwp_files)}")

