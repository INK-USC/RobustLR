from helper import *


class DataModule(pl.LightningDataModule):

	def __init__(self, dataset, train_dataset, dev_dataset, test_dataset, arch, train_batch_size=32, eval_batch_size=32,\
					num_workers=10, pad_idx=0, ood_test_dataset=''):
		super().__init__()
		self.p                  = types.SimpleNamespace()
		self.p.dataset          = dataset
		self.p.train_dataset    = train_dataset		# used in load_dataset()
		self.p.dev_dataset      = dev_dataset		# used in load_dataset()
		self.p.test_dataset     = test_dataset		# used in load_dataset()
		self.p.ood_test_dataset = ood_test_dataset
		self.p.actual_arch      = arch
		self.p.arch             = arch
		self.p.train_batch_size = train_batch_size
		self.p.eval_batch_size  = eval_batch_size
		self.p.num_workers      = num_workers
		self.p.pad_idx          = pad_idx

	def load_dataset(self, split, arch):
		dataset = ddict(list)

		if arch.startswith('t5'):
			all_folders = [f'../data/processed/{x}/t5_large/{split}/' for x in getattr(self.p, f'{split}_dataset').split(',')]
			print(f'allfolders for {split}: {all_folders}')
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				for folder in all_folders:
					with open(folder + f'{key}.pkl', 'rb') as f:
						try:
							with open(folder + f'{key}.pkl', 'rb') as f:
								tmp          = pickle.load(f)
								dataset[key] = dataset[key] + tmp
						except Exception as e:
							print(f'Missing Key {key}')
							assert key == 'theory_id'
		else:
			all_folders = [f'../data/processed/{x}/{arch}/{split}/' for x in getattr(self.p, f'{split}_dataset').split(',')]
			print(f'allfolders for {split}: {all_folders}')
			for key in ['input_ids', 'label', 'ctx_ids', 'stmt_ids', 'has_neg', 'theory_id']:
				for folder in all_folders:
					try:
						with open(folder + f'{key}.pkl', 'rb') as f:
							tmp          = pickle.load(f)
							dataset[key] = dataset[key] + tmp
					except Exception as e:
						print(f'Missing Key {key}')
						assert key == 'theory_id'

		dataset = dict(dataset)

		return dataset

	def load_ood_dataset(self, arch):
		dataset = ddict(list)
		if arch.startswith('t5'):
			all_folders = [f'../data/processed/{self.p.ood_test_dataset}/t5_large/test/']
			print('OOD folders', all_folders)
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				for folder in all_folders:
					with open(folder + f'{key}.pkl', 'rb') as f:
						try:
							with open(folder + f'{key}.pkl', 'rb') as f:
								tmp          = pickle.load(f)
								dataset[key] = dataset[key] + tmp
						except Exception as e:
							print(f'Missing Key {key}')
							assert key == 'theory_id'

		elif arch == 'roberta_large_race':
			all_folders = [f'../data/processed/{self.p.ood_test_dataset}/{arch}/test/']
			print('OOD folders', all_folders)
			for key in ['input_ids', 'label', 'ctx_ids', 'stmt_ids', 'has_neg', 'theory_id']:
				for folder in all_folders:
					try:
						with open(folder + f'{key}.pkl', 'rb') as f:
							tmp          = pickle.load(f)
							dataset[key] = dataset[key] + tmp
					except Exception as e:
						print(f'Missing Key {key}')
						assert key == 'theory_id'

		return dict(dataset)

	def setup(self, stage=None, splits='all'):
		self.data = ddict(list)
		if splits == 'all':
			splits = ['train', 'dev', 'test']

		for split in splits:
			if self.p.arch == 't5_large':
				self.data[split] = GenerativeDataset(self.load_dataset(split, self.p.arch), self.p.pad_idx)
			elif self.p.arch == 'roberta_large_race':
				self.data[split] = DiscriminativeDataset(self.load_dataset(split, self.p.arch), self.p.pad_idx)

		if self.p.ood_test_dataset != '':
			if self.p.arch == 't5_large':
				self.data['ood_test'] = GenerativeDataset(self.load_ood_dataset(self.p.arch), self.p.pad_idx, iid=False)
			elif self.p.arch == 'roberta_large_race':
				self.data['ood_test'] = DiscriminativeDataset(self.load_ood_dataset(self.p.arch), self.p.pad_idx, iid=False)

	def train_dataloader(self, shuffle=True):
		return DataLoader(
					self.data['train'],
					batch_size=self.p.train_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['train'].collater,
					shuffle=shuffle,
					pin_memory=True
				)

	def val_dataloader(self):
		return DataLoader(
					self.data['dev'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['dev'].collater,
					pin_memory=True
				)

	def test_dataloader(self, split='test'):
		return DataLoader(
					self.data[split],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['test'].collater,
					pin_memory=True
				)

	@staticmethod
	def add_data_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument("--dataset", 		 				    type=str)
		parser.add_argument("--train_dataset",	    default='', 	type=str)
		parser.add_argument("--dev_dataset",	    default='', 	type=str)
		parser.add_argument("--test_dataset",	    default='', 	type=str)
		parser.add_argument("--ood_test_dataset",	default='', 	type=str)
		parser.add_argument("--num_workers", 	    default=10, 	type=int)
		return parser


class DiscriminativeDataset(Dataset):

	def __init__(self, dataset, pad_idx, iid=True):
		self.data    = dataset
		self.pad_idx = pad_idx
		self.iid     = iid

	def __len__(self):
		return len(self.data['label'])

	def __getitem__(self, idx):
		theory_id = self.data['theory_id'][idx] if 'theory_id' in self.data else 'None'
		lbl       = self.data['label'][idx]

		item = {
			'sent'     : torch.LongTensor(self.data['input_ids'][idx]),
			'ctx'      : torch.LongTensor(self.data['ctx_ids'][idx]),
			'stmt'     : torch.LongTensor(self.data['stmt_ids'][idx]),
			'lbl'      : torch.LongTensor([lbl]),
			'ctx_len'  : torch.LongTensor([len(self.data['ctx_ids'][idx])]),
			'stmt_len' : torch.LongTensor([len(self.data['stmt_ids'][idx])]),
			'has_neg'  : torch.LongTensor([1 if self.data['has_neg'][idx] == 'True' else 0]),
			'iid'      : self.iid,
			'theory_id': theory_id,
		}

		return item

	def collater(self, items):
		all_sents = pad_sequence([x['sent'] for x in items], batch_first=True, padding_value=self.pad_idx)

		batch = {
			'all_sents' : all_sents,
			'all_ctxs'  : pad_sequence([x['ctx'] for x in items], batch_first=True, padding_value=self.pad_idx),
			'all_stmts' : pad_sequence([x['stmt'] for x in items], batch_first=True, padding_value=self.pad_idx),
			'all_lbls'  : torch.cat([x['lbl'] for x in items]),
			'ctx_lens'  : torch.cat([x['ctx_len'] for x in items]),
			'stmt_lens' : torch.cat([x['stmt_len'] for x in items]),
			'attn_mask' : (all_sents != self.pad_idx).long(),
			'has_neg'   : torch.cat([x['has_neg'] for x in items]),
			'iid'       : [x['iid'] for x in items],
			'theory_ids': [x['theory_id'] for x in items],
		}

		return batch


class GenerativeDataset(Dataset):

	def __init__(self, dataset, pad_idx, iid=True):
		self.data    = dataset
		self.pad_idx = pad_idx
		self.iid     = iid

	def __len__(self):
		return len(self.data['input_ids'])

	def __getitem__(self, idx):
		theory_id = self.data['theory_id'][idx] if 'theory_id' in self.data else 'None'

		item = {
			'input'    : torch.LongTensor(self.data['input_ids'][idx]),
			'output'   : torch.LongTensor(self.data['output_ids'][idx]),
			'has_neg'  : torch.LongTensor([1 if self.data['has_neg'][idx] == 'True' else 0]),
			'iid'      : self.iid,
			'theory_id': theory_id,
		}

		return item

	def collater(self, items):
		all_inps        = pad_sequence([x['input'] for x in items], batch_first=True, padding_value=self.pad_idx)
		all_outs        = pad_sequence([x['output'] for x in items], batch_first=True, padding_value=self.pad_idx)

		labels = all_outs.clone()
		labels[labels == self.pad_idx] = -100

		batch = {
			'all_inps'         : all_inps,
			'attn_mask'        : (all_inps != self.pad_idx).long(),
			'labels'           : labels,
			'labels_for_decode': all_outs,
			'has_neg'          : torch.cat([x['has_neg'] for x in items]),
			'iid'              : [x['iid'] for x in items],
			'theory_ids'       : [x['theory_id'] for x in items],
		}

		return batch
