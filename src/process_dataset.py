from helper import *

csv.field_size_limit(sys.maxsize)
total_instances, truncated_instances = 0, 0


class LogicalInstance:

	def __init__(self, context, statement, label, proofs, proof_deps, has_neg, theory_id):
		self.context     = context
		self.statement   = statement
		self.label       = label
		self.proofs      = proofs
		self.proof_deps  = proof_deps
		self.has_neg     = has_neg
		self.theory_id   = theory_id

	@classmethod
	def from_csv(cls, row):
		proof_deps = row[4]
		proof_deps = list(map(int, proof_deps.split(',')))

		if len(row) == 7:
			return LogicalInstance(row[0], row[1], int(row[2]), row[3], proof_deps, row[5], row[6])
		else:
			return LogicalInstance(row[0], row[1], int(row[2]), row[3], proof_deps, row[5], '')

	def tokenize_ptlm(self, tokenizer):
		global total_instances, truncated_instances

		input_tokens = tokenizer.cls_token + self.context + tokenizer.sep_token + self.statement + tokenizer.sep_token
		input_ids    = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_tokens))
		ctx_ids      = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.cls_token + self.context + tokenizer.sep_token))
		stmt_ids     = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.cls_token + self.statement + tokenizer.sep_token))

		total_instances += 1

		if len(input_ids) > tokenizer.model_max_length:
			return None

		return input_ids[:tokenizer.model_max_length], ctx_ids[:tokenizer.model_max_length], stmt_ids

	def tokenize_ptlm_t5(self, tokenizer):
		global total_instances, truncated_instances

		input_str = '$answer$ ; $question$ = '
		self.statement = self.statement[:-1] + '?'
		input_str += self.statement + ' ; $context$ ='
		input_str_split = input_str
		ctx = ''
		for sent in self.context.split('. '):
			sent = sent + '.'
			ctx += f' {sent}'
		input_str_split += ctx
		input_str_split = input_str_split[:-1]

		input_str_nltk = input_str
		ctx = ''
		for sent in sent_tokenize(self.context):
			ctx += f' {sent}'
		input_str_nltk += ctx

		assert input_str_split == input_str_nltk
		input_str = input_str_nltk
		input_str += tokenizer.eos_token
		input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))

		label = map_int_to_str(self.label)
		output_ids = tokenizer(f'$answer$ = {label}').input_ids

		total_instances += 1

		if len(input_ids) > tokenizer.model_max_length:
			return None

		return input_ids[:tokenizer.model_max_length], output_ids[:tokenizer.model_max_length]

	def tokenize(self, tokenizer, arch):
		if arch == 'roberta_large_race':
			return self.tokenize_ptlm(tokenizer)
		if arch == 't5_large':
			return self.tokenize_ptlm_t5(tokenizer)


def map_int_to_str(label):
	if label == 0:
		label = 'False'
	elif label == 1:
		label = 'True'
	elif label == 2:
		label = 'Nothing'
	return label

def get_inp_fname(args, split):
	return f'../data/{args.dataset}/{split}.csv'

def get_out_fname(args, split, key=None, dir_name=False):
	dataset = f'{args.dataset.split("/")[-1]}'

	if dir_name:
		return f'../data/processed/{dataset}/{args.arch}/{split}/'
	else:
		return f'../data/processed/{dataset}/{args.arch}/{split}/{key}.pkl'

def main(args):
	if args.eval:
		test_length = 20000
	else:
		train_length = 50000
		dev_length   = 10000
		test_length  = 10000

	# load tokenizer
	if args.arch == 'roberta_large_race':
		tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
	elif args.arch == 't5_large':
		tokenizer = AutoTokenizer.from_pretrained('t5-large')
	else:
		print('Token type ids not implemented in tokenize call, will not work for bert models')
		import pdb; pdb.set_trace()
		raise NotImplementedError

	# load data
	for split in ['train', 'dev', 'test']:

		print(f'Processing {split} split...')

		# make folder if not exists
		print(f'Creating directory {get_out_fname(args, split, dir_name=True)}')
		pathlib.Path(get_out_fname(args, split, dir_name=True)).mkdir(exist_ok=True, parents=True)

		data = ddict(list)

		if args.arch == 'roberta_large_race':
			# load the relevant file and select all the data
			with open(get_inp_fname(args, split)) as f:
				reader = csv.reader(f)
				for row in tqdm(reader):
					instance = LogicalInstance.from_csv(row)
					output   = instance.tokenize(tokenizer, args.arch)
					if output is not None:
						data['input_ids'].append(output[0])
						data['ctx_ids'].append(output[1])
						data['stmt_ids'].append(output[2])
						data['label'].append(instance.label)
						data['deps'].append(instance.proof_deps)
						data['has_neg'].append(instance.has_neg)
						data['theory_id'].append(instance.theory_id)

			data = dict(data)

			# write the data in pickle format to processed folder
			for key in ['input_ids', 'ctx_ids', 'stmt_ids', 'label', 'deps', 'has_neg', 'theory_id']:
				print(f'Contains {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				if split == 'test':
					data[key] = data[key][:test_length]
				elif split == 'train' and not args.eval:
					data[key] = data[key][:train_length]
				elif split == 'dev' and not args.eval:
					data[key] = data[key][:dev_length]
				print(f'Dumping {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				with open(get_out_fname(args, split, key=key, dir_name=False), 'wb') as f:
					pickle.dump(data[key], f)

		elif args.arch == 't5_large':
			# load the relevant problog file and select all the data
			with open(get_inp_fname(args, split)) as f:
				reader = csv.reader(f)
				for row in tqdm(reader):
					instance = LogicalInstance.from_csv(row)
					output = instance.tokenize(tokenizer, args.arch)
					if output is not None:
						data['input_ids'].append(output[0])
						data['output_ids'].append(output[1])
						data['has_neg'].append(instance.has_neg)
						data['theory_id'].append(instance.theory_id)

			data = dict(data)

			# write the data in pickle format to processed folder
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				print(f'Contains {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				if split == 'test':
					data[key] = data[key][:test_length]
				elif split == 'train' and not args.eval:
					data[key] = data[key][:train_length]
				elif split == 'dev' and not args.eval:
					data[key] = data[key][:dev_length]

				print(f'Dumping {len(data[key])} lines for dataset: {args.dataset} split: {split} key: {key}')
				with open(get_out_fname(args, split, key=key, dir_name=False), 'wb') as f:
					pickle.dump(data[key], f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocessing the dataset')

	parser.add_argument('--dataset')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--arch', default='roberta_large_race', choices=['roberta_large_race', 't5_large'])

	args = parser.parse_args()

	main(args)
