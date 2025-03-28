import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Bio import SeqIO

# 定义 StudentModelWithMapping 类
class StudentModelWithMapping(torch.nn.Module):
    def __init__(self, student_model, student_hidden_size=320, teacher_hidden_size=1280):
        super(StudentModelWithMapping, self).__init__()
        self.student_model = student_model
        self.mapping = torch.nn.Linear(student_hidden_size, teacher_hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        student_features = hidden_states[-1].mean(dim=1)  # [batch_size, hidden_dim]
        mapped_features = self.mapping(student_features)  # [batch_size, teacher_hidden_size]
        return logits, mapped_features

# Classification head
class MLP(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, num_classes=2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 定义数据集
class FeatureExtractionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# 特征提取函数
def extract_features(model, dataloader, device="cpu"):
    all_original_features = []
    all_mapped_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            _, mapped_features = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = model.student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            original_features = hidden_states[-1].mean(dim=1)

            all_original_features.append(original_features.cpu().numpy())
            all_mapped_features.append(mapped_features.cpu().numpy())

    all_original_features = np.concatenate(all_original_features, axis=0)
    all_mapped_features = np.concatenate(all_mapped_features, axis=0)
    return all_original_features, all_mapped_features

# 读取FASTA文件并进行分词
def preprocess_fasta(fasta_file, tokenizer, max_length=100):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    encodings = tokenizer(sequences, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return encodings

# 主函数
def main(args):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("./student_model")
    base_student_model = AutoModelForSequenceClassification.from_pretrained("./student_model")
    student_model = StudentModelWithMapping(
        student_model=base_student_model,
        student_hidden_size=base_student_model.config.hidden_size,
        teacher_hidden_size=1280
    )
    student_model.load_state_dict(torch.load("./student_model_05/student_model_with_mapping.pt", map_location="cpu"))
    student_model.eval()

    # 预处理数据
    encodings = preprocess_fasta(args.input_file, tokenizer)
    dataset = FeatureExtractionDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 提取特征
    original_features, mapped_features = extract_features(student_model, dataloader, device="cpu")

    # 保存特征
    np.save(args.output_original, original_features)
    np.save(args.output_mapped, mapped_features)
    print(f"Features saved: {args.output_original}, {args.output_mapped}")

    # 是否进行预测
    if args.make_prediction:
        print("Making predictions with MLP...")

        # 加载 MLP 模型
        mlp_model = MLP(input_size=1280, hidden_size=512, num_classes=2)
        mlp_model.load_state_dict(torch.load("./student_model_05/Classification_Head.pth", map_location="cpu"))
        mlp_model.eval()

        mapped_features_tensor = torch.tensor(mapped_features, dtype=torch.float32)
        with torch.no_grad():
            logits = mlp_model(mapped_features_tensor)  # 获取分类头输出
            probabilities = torch.nn.functional.softmax(logits, dim=1)  # 转换为概率
            predictions = torch.argmax(probabilities, dim=1)  # 获取预测类别

        # 读取 FASTA IDs
        fasta_ids = [record.id for record in SeqIO.parse(args.input_file, "fasta")]

        # 确保长度匹配
        assert len(fasta_ids) == len(predictions), "FASTA IDs and predictions do not match!"

        # 保存预测结果
        with open(args.output_predictions, "w") as f:
            for fasta_id, pred_class, prob in zip(fasta_ids, predictions, probabilities):
                prob_str = ",".join(map(str, prob.tolist()))  # 将概率转为逗号分隔的字符串
                f.write(f"{fasta_id}\t{pred_class.item()}\t{prob_str}\n")

        print(f"Predictions saved to {args.output_predictions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a student model and classify with MLP.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output_original", type=str, required=True, help="Path to save the original 320-dim features.")
    parser.add_argument("--output_mapped", type=str, required=True, help="Path to save the mapped 1280-dim features.")
    parser.add_argument("--output_predictions", type=str, help="Path to save the predictions if make_prediction is enabled.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--make_prediction", action='store_true', help="Flag to enable prediction with MLP.")
    args = parser.parse_args()

    main(args)