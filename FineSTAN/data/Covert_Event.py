"""
Author ：Fan wang
Time ：2026. 1.7
"""
# 1. Read the event.tsv file
input_path = r'D:\Handwriting_Imagery_Raw_Dara\CCS-SV-HI-EEG_BIDS\sub-03\ses-01\eeg\sub-03_ses-01_task-CCSHI_events.tsv'  # Replace with the actual path  sub-01_ses-02_task-SV-HI_events.tsv'
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2. Parse the header to find the column indices of "sample" and "value"
header = lines[0].strip().split("\t")
sample_col = header.index("sample")
value_col = header.index("value")

# 3. Calculate the maximum width of the first column (critical for alignment)
# 提取所有第一列数据，计算最大字符长度
samples = [line.strip().split("\t")[sample_col] for line in lines[1:]]
max_length = max(len(s) for s in samples)  # 第一列的最大宽度（如7位数字则为7）

# 4. Extract data and write to a new file with strict alignment for the second column
output_path = r'D:\Handwriting_Imagery_Raw_Dara\CCS_HI_Event_Txt\A03T_Event.txt'
with open(output_path, "w", encoding="utf-8") as f_out:
    for line in lines[1:]:  # Skip the header
        parts = line.strip().split("\t")
        sample = parts[sample_col]
        value = parts[value_col]

        # 关键：第一列左对齐并强制占用max_length宽度，加4个固定空格分隔，确保第二列对齐
        # < 表示左对齐，{max_length} 强制第一列宽度，后续空格确保第二列起始位置一致
        f_out.write(f"{sample:<{max_length}}    {value}\n")

print(f"Conversion completed! The result has been saved to {output_path}")
print(f"First column fixed width: {max_length} characters. Second column aligned.")