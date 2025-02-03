from collections import Counter
from collections import defaultdict
import gc

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
from tqdm.notebook import tqdm
from datasets import concatenate_datasets, Dataset as HFDataset


class CyclicTextDataset(Dataset):
    def __init__(self, encoded_texts, needed_augmentations):
        """
        encoded_texts: A dictionary containing {'input_ids': tensor, 'attention_mask': tensor}
        needed_augmentations: Number of samples to generate
        """
        super().__init__()
        self.encoded_texts = encoded_texts
        self.needed_augmentations = needed_augmentations
        self.num_samples = encoded_texts["input_ids"].shape[0]

    def __len__(self):
        return self.needed_augmentations

    def __getitem__(self, idx):
        idx = idx % self.num_samples  # Cycle through original samples
        return {k: self.encoded_texts[k][idx] for k in self.encoded_texts}


class TranslationAugmentor:
    def __init__(self, src="en", mid="fr", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.src = src
        self.mid = mid
        self.model_forward, self.tokenizer_forward = self.load_translation_model(
            src, mid
        )
        self.model_backward, self.tokenizer_backward = self.load_translation_model(
            mid, src
        )

    def load_translation_model(self, src, tgt):
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(self.device)
        model.eval()
        return model, tokenizer

    # Don't need gradients for translation. Turn off autograd for faster inference
    @torch.no_grad()
    def translate(
        self,
        model,
        encoded_texts,
        needed_augmentations,
        src_lang,
        tgt_lang,
        batch_size,
        max_workers,
    ):
        """
        Translate encoded texts from src_lang to tgt_lang.
        """
        model = model.to(self.device)
        output = []

        # Use adaptive batch size to speed up small input inference
        batch_size = min(batch_size, len(encoded_texts["input_ids"]))

        # Use custom pytorch dataset for cyclic sampling
        encoded_ds = CyclicTextDataset(encoded_texts, needed_augmentations)

        # CPU pre-fetching for faster data loading
        num_workers = min(max_workers, torch.get_num_threads())

        # Use DataLoader for steady GPU memory consumption
        data_loader = DataLoader(
            encoded_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Progress bar for visualization
        pbar = tqdm(
            data_loader,
            desc=f"Translating {src_lang} → {tgt_lang}",
            unit=f"batch",
            unit_scale=True,
            leave=False,
        )

        # Inference loop
        for batch_encoded in pbar:
            batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
            translated = model.generate(**batch_encoded)
            output.extend(translated.cpu().tolist())

            # Cleanup memory manually because the MarianMT doesn't do that automatically for inference
            del batch_encoded, translated
            torch.cuda.empty_cache()
            gc.collect() 

        return output

    def translate_pipeline(
        self,
        model,
        tokenizer,
        texts,
        needed_augmentations,
        src_lang,
        tgt_lang,
        batch_size,
        max_workers,
    ):
        # Tokenize once before inference for faster inference
        encoded_texts = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )

        # Run inference
        translated = self.translate(
            model,
            encoded_texts,
            needed_augmentations,
            src_lang,
            tgt_lang,
            batch_size,
            max_workers,
        )

        # Cleanup memory manually because the MarianMT doesn't do that automatically for inference
        del encoded_texts
        torch.cuda.empty_cache()
        gc.collect()

        # Return decoded output
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def demo(self, texts, batch_size=32, max_workers=4):
        # Foward translation
        intermediate_texts = self.translate_pipeline(
            self.model_forward,
            self.tokenizer_forward,
            texts,
            len(texts),
            self.src,
            self.mid,
            batch_size,
            max_workers,
        )

        # Backward translation
        augmented_texts = self.translate_pipeline(
            self.model_backward,
            self.tokenizer_backward,
            intermediate_texts,
            len(texts),
            self.mid,
            self.src,
            batch_size,
            max_workers,
        )

        return augmented_texts

    def augment_dataset(
        self, dataset, sentence_key, label_key, batch_size=32, max_workers=4
    ):
        # Extract data from hf dataset
        labels_list = dataset[label_key]
        texts_list = dataset[sentence_key]
        num_rows = len(dataset)

        # Count how many samples per label
        label_counts = Counter(labels_list)

        # Augment to match the count of dominant labels
        target_sample_size = max(label_counts.values())

        # Group texts by label
        texts_by_label = defaultdict(list)
        for i in range(num_rows):
            lbl = labels_list[i]
            txt = texts_list[i]
            texts_by_label[lbl].append(txt)

        # Collect augmented rows
        augmented_samples = []

        pbar = tqdm(label_counts.items(), desc="Augmenting Labels", leave=True)
        for label, count in pbar:
            pbar.set_postfix({"Current Label": label})
            if count >= target_sample_size:
                continue  # no need to augment this label

            # Count the number of augmented data needed
            needed_augmentations = target_sample_size - count
            text_list = texts_by_label[label]

            # Foward translation
            intermediate_texts = self.translate_pipeline(
                self.model_forward,
                self.tokenizer_forward,
                text_list,
                needed_augmentations,
                self.src,
                self.mid,
                batch_size,
                max_workers,
            )

            # Backward translation
            augmented_texts = self.translate_pipeline(
                self.model_backward,
                self.tokenizer_backward,
                intermediate_texts,
                needed_augmentations,
                self.mid,
                self.src,
                batch_size,
                max_workers,
            )

            # Collect augmented samples
            for aug_text in augmented_texts:
                augmented_samples.append({sentence_key: aug_text, label_key: label})

        # Adding augmented data to dataset
        if augmented_samples:
            augmented_dataset = HFDataset.from_list(augmented_samples)
            dataset = concatenate_datasets([dataset, augmented_dataset])

        return dataset.shuffle()
