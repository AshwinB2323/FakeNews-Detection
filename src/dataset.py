import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FakeNewsDataset(Dataset):
	def __init__(self, file_path):
		self.data = pd.read_csv(file_path)
		
		print(f" Debug: Columns in {file_path}: {self.data.columns.tolist()}")

		if 'title' in self.data.columns:
			self.data.rename(columns={'title': 'headline'}, inplace = True)
		if 'content' in self.data.columns:
			self.data.rename(columns={'content': 'text'}, inplace = True)
		if 'tweet' in self.data.columns:
			self.data.rename(columns={'tweet': 'text'}, inplace = True)
		if 'index' in self.data.columns and 'label' in self.data.columns:
			self.data.rename(columns={'index': 'text'}, inplace = True)

		self.is_news = 'headline' in self.data.columns

		required_columns = ['headline', 'text', 'label'] if self.is_news else ['text', 'label']
		
		missing_columns = [col for col in required_columns if col not in self.data.columns]

		if missing_columns:
			raise KeyError(f"Missing columns: {missing_columns}, Available columns: {self.data.columns.tolist()}")


		self.data = self.data[required_columns]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		sample = {
			'text': row['text'],
			'label': 1 if row['label'].lower() == 'real' else 0
		}
		if self.is_news:
			sample['headline'] = row['headline']
		return sample

def get_data_loader(file_path, batch_size = 16, shuffle = True):
	dataset = FakeNewsDataset(file_path)
	return DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)

