import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from evaluate import load


class CustomBertModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super().__init__()
        # Use pre-trained bert for sequence classification as base model
        pre_trained_model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        self.bert = pre_trained_model.base_model

        # Use the correspondent tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Freeze base model parameters because I only have a small dataset and a laptop
        for param in self.bert.parameters():
            param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size, bias=True)

        # Key for nonlinear classification
        self.activation = nn.ReLU()

        # Dropout layer to prevent overfitting on a small dataset
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)

        # CrossEntropyLoss is used for classification loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get outputs from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the [CLS] embeddings for classification
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)

        # Feed embeddings to the classifier with 2 layers
        x = self.pre_classifier(pooled_output)
        x = self.activation(x)

        # Dropout layer prevents overfitting
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # Return (loss, logits) if labels were provided, else just logits
        return (loss, logits) if loss is not None else logits

    def train_loop(
        self,
        train_dataset,
        sentence_key,
        label_key,
        eval_dataset=None,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        # Convert HF datasets to PyTorch DataLoader
        train_loader = self._create_dataloader(
            train_dataset, sentence_key, label_key, batch_size, shuffle=True
        )
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = self._create_dataloader(
                eval_dataset, sentence_key, label_key, batch_size, shuffle=False
            )

        # Metrics (accuracy & F1). Common for classification tasks
        accuracy_metric = load("accuracy")
        f1_metric = load("f1")

        # Save history for plotting
        train_loss_history = []
        val_accuracy_history = []
        val_f1_history = []

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            for batch in pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                loss, logits = self(input_ids, attention_mask, labels=labels)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            print(f"Train Loss: {avg_train_loss:.4f}")

            # Model valuation
            if eval_loader is not None:
                self.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in eval_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        # forward
                        _, logits = self(input_ids, attention_mask, labels=labels)
                        preds = torch.argmax(logits, dim=-1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # Compute metrics
                acc = accuracy_metric.compute(
                    predictions=all_preds, references=all_labels
                )["accuracy"]
                f1_macro = f1_metric.compute(
                    predictions=all_preds, references=all_labels, average="macro"
                )["f1"]
                val_accuracy_history.append(acc)
                val_f1_history.append(f1_macro)
                print(f"Eval Accuracy: {acc:.4f}, Eval F1 (macro): {f1_macro:.4f}")

        print("Training complete.")
        return train_loss_history, val_accuracy_history, val_f1_history

    def _create_dataloader(
        self, dataset, sentence_key, label_key, batch_size, shuffle=False
    ):

        def encode_batch(batch):
            encoding = self.tokenizer(
                batch[sentence_key],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            encoding["labels"] = batch[label_key]
            return encoding

        # Tokenize for the batch
        dataset = dataset.map(encode_batch, batched=True)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def predict(self, texts, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.eval()
        self.to(device)

        # Tokenize (batched)
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            # forward
            logits = self(input_ids=input_ids, attention_mask=attention_mask)
            # If the forward returns (loss, logits), handle that:
            if isinstance(logits, tuple):
                logits = logits[1]

        # argmax to get predicted class IDs
        preds = torch.argmax(logits, dim=-1)
        return preds.cpu().tolist()
