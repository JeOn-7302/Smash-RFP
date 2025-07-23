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
        
# print(f"총 파일 수: {len(os.listdir(data_dir))}")
# print(f"PDF 파일 수: {len(pdf_files)}")
# print(f"HWP 파일 수: {len(hwp_files)}")

# import fitz  # PyMuPDF

# for filename in pdf_files:
#     filepath = os.path.join(data_dir, filename)
#     try:
#         doc = fitz.open(filepath)
#         text_len = sum(len(page.get_text().strip()) for page in doc)
#         print(f"{filename} - 추출된 텍스트 길이: {text_len}")
#     except Exception as e:
#         print(f"{filename} - 오류 발생: {e}")

# import sys
# sys.path.append('/home/daeseok/pyhwp')

import hwp5.hwp5txt

with open("/home/daeseok/AI-Engineer/data/고양도시관리공사_관산근린공원 다목적구장 홈페이지 및 회원 통합운영.hwp", "rb") as f:
    text = hwp5.hwp5txt.extract_text(f)
    
# try:
#     import hwp5
#     print("✅ hwp5 모듈이 정상적으로 import 되었습니다.")
# except ImportError as e:
#     print("❌ hwp5 모듈 import 실패:", e)
        
# from pyhwp.hwp5.hwp5file import HWP5File

# for filename in hwp_files:
#     filepath = os.path.join(data_dir, filename)
#     try:
#         doc = hwp5.HWP5File(filepath)
#         bodytext = ""
#         for section in doc.bodytext.section_list:
#             for para in section.paragraph_list:
#                 bodytext += para.text + "\n"
#         print(f"{filename} - 추출된 텍스트 길이: {len(bodytext)}")
#     except Exception as e:
#         print(f"{filename} - ❌ 오류 발생: {e}")

# PDF 텍스트 추출


# %%
a = 10
# %%

