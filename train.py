# main_training.py

# ==============================================================================
# 1. Required Imports
# ==============================================================================
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, default_data_collator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- IMPORT THE CUSTOM LOSS FUNCTIONS FROM losses.py ---
from distllation import distillation_loss, feature_alignment_loss

# ==============================================================================
# 2. Student Model Architecture Definition
# ==============================================================================
class StudentModelWithMapping(nn.Module):
    """
    Defines the student model with an added feature-mapping layer.
    """
    def __init__(self, student_model, student_hidden_size, teacher_hidden_size, num_labels):
        super(StudentModelWithMapping, self).__init__()
        self.student_model = student_model
        self.mapping = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(teacher_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        student_features = hidden_states[-1].mean(dim=1)
        mapped_features = self.mapping(student_features)
        logits = self.classifier(mapped_features)
        return logits, mapped_features

# ==============================================================================
# 3. Custom Dataset Class
# ==============================================================================
class DistillationDataset(Dataset):
    """
    Custom PyTorch Dataset for knowledge distillation.
    """
    def __init__(self, encodings, labels, teacher_logits, teacher_features):
        self.encodings = encodings
        self.labels = labels
        self.teacher_logits = teacher_logits
        self.teacher_features = teacher_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['teacher_logits'] = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
        item['teacher_features'] = torch.tensor(self.teacher_features[idx], dtype=torch.float32)
        return item

# ==============================================================================
# 4. Main Training Function
# ==============================================================================
def main(args):
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data and Tokenizer ---
    print("Loading data and tokenizer...")
    train_df = pd.read_csv(args.train_csv)
    teacher_logits = np.load(args.teacher_logits)
    teacher_features = np.load(args.teacher_features)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # --- Data Preprocessing ---
    print("Preprocessing data...")
    train_sequences = train_df['seq'].tolist()
    train_labels = train_df['label'].tolist()
    train_encodings_dict = tokenizer(train_sequences, padding='max_length', truncation=True, max_length=args.max_length, return_tensors="pt")
    
    train_dataset = DistillationDataset(
        encodings=train_encodings_dict,
        labels=train_labels,
        teacher_logits=teacher_logits,
        teacher_features=teacher_features
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    # --- Initialize Model ---
    print("Initializing student model...")
    base_student_model = AutoModel.from_pretrained(args.student_model_path)
    
    student_model = StudentModelWithMapping(
        student_model=base_student_model,
        student_hidden_size=base_student_model.config.hidden_size,
        teacher_hidden_size=teacher_features.shape[1],
        num_labels=len(np.unique(train_labels))
    ).to(device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)

    # --- Start Training ---
    print("Starting distillation training...")
    student_model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            teacher_logits_batch = batch['teacher_logits'].to(device)
            teacher_features_batch = batch['teacher_features'].to(device)

            student_logits, student_mapped_features = student_model(input_ids=input_ids, attention_mask=attention_mask)

            # --- CALCULATE AND COMBINE THE TWO DISTILLATION LOSSES ---
            loss_logits = distillation_loss(student_logits, teacher_logits_batch, labels, args.alpha, args.temperature)
            loss_features = feature_alignment_loss(student_mapped_features, teacher_features_batch)
            
            # Total Loss = Response-Based Loss + (beta * Feature-Based Loss)
            loss = loss_logits + args.beta * loss_features

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} finished. Average Training Loss: {avg_loss:.4f}")

    # --- Save the Final Model After Training ---
    print("\nTraining complete. Saving final model...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.save(student_model.state_dict(), os.path.join(args.output_dir, "student_model_with_mapping.pt"))
    student_model.student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Final student model and tokenizer saved to '{args.output_dir}'.")

# ==============================================================================
# 5. Command-Line Argument Parser
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training Script for ESM Models")

    # --- File Path Arguments ---
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--teacher_logits", type=str, required=True, help="Path to the .npy file with teacher's logits.")
    parser.add_argument("--teacher_features", type=str, required=True, help="Path to the .npy file with teacher's features.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the pretrained tokenizer directory.")
    parser.add_argument("--student_model_path", type=str, required=True, help="Path to the base student model directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained student model.")

    # --- Training Hyperparameters ---
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    
    # --- Distillation Hyperparameters ---
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the hard label loss (cross-entropy).")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature for softening logits in distillation.")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for the feature alignment loss (MSE).")

    args = parser.parse_args()
    main(args)
