from helper import *
from basemodel import BaseModel
import eval_consistency_f1 as eval_func



class LRGenerative(BaseModel):
	def __init__(self, arch='t5_large', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,\
					optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, gpus=1,\
					hf_name='t5-large',save_dir=None, num_classes=3, compute_metrics =False):
		super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)

		self.compute_metrics = compute_metrics
		self.save_hyperparameters()

		self.p                         = types.SimpleNamespace()
		self.p.arch                    = arch
		self.p.train_batch_size        = train_batch_size
		self.p.eval_batch_size         = eval_batch_size
		self.p.accumulate_grad_batches = accumulate_grad_batches
		self.p.learning_rate           = learning_rate
		self.p.max_epochs              = max_epochs
		self.p.optimizer               = optimizer
		self.p.adam_epsilon            = adam_epsilon
		self.p.weight_decay            = weight_decay
		self.p.lr_scheduler            = lr_scheduler
		self.p.warmup_updates          = warmup_updates
		self.p.gpus                    = gpus
		self.p.save_dir                = save_dir
		self.p.hf_name    			   = hf_name
		self.p.num_classes             = num_classes

		self.reasoner          = T5ForConditionalGeneration.from_pretrained(hf_name)
		self.tokenizer         = AutoTokenizer.from_pretrained(hf_name)
		self.generator_options = {'min_length': 1, 'max_length': 128, 'num_beams': 1, 'num_return_sequences': 1, 'do_sample': False, 'top_k': 50, 'top_p': 1.0,\
									'temperature': 1.0, 'length_penalty': 1.0, 'repetition_penalty': 1.0}

	def forward(self, batch):
		outputs = self.reasoner(input_ids=batch['all_inps'], attention_mask=batch['attn_mask'], labels=batch['labels'])
		return outputs

	def predict(self, batch):
		output_ids = self.reasoner.generate(batch['all_inps'], **self.generator_options)
		output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

		return output_str

	def calc_acc(self, preds, targets):
		return 100 * (preds == targets).float().mean()

	def map_str_to_int(self, labels, num_classes):
		int_labels = []
		for label in labels:
			if '$answer$ = False' == label:
				int_labels.append(0)
			elif '$answer$ = True' == label:
				int_labels.append(1)
			elif '$answer$ = Nothing' == label and num_classes == 3:
				int_labels.append(2)
			else:
				int_labels.append(3)

		return torch.LongTensor(int_labels)

	def run_step(self, batch, split):
		out  = self(batch)
		loss = out.loss

		if split == 'train':
			self.log(f'train_loss_step', loss.item(), prog_bar=True)

			return {'loss': loss}
		else:
			targets     = self.tokenizer.batch_decode(batch['labels_for_decode'], skip_special_tokens=True)
			int_targets = self.map_str_to_int(targets, self.p.num_classes)
			assert 3 not in int_targets

			preds     = self.predict(batch)
			int_preds = self.map_str_to_int(preds, self.p.num_classes)
			acc       = self.calc_acc(int_preds, int_targets)

			self.log(f'{split}_loss_step', loss.item(), prog_bar=True, sync_dist=True)
			self.log(f'{split}_acc_step', acc, prog_bar=True, sync_dist=True)

			return {'loss': loss, 'acc': acc, 'preds': int_preds, 'targets': int_targets, 'neg_indicator': batch['has_neg'], 'iid': batch['iid'], 'theory_ids': batch['theory_ids']}

	def aggregate_epoch(self, outputs, split):

		loss = torch.stack([x['loss'] for x in outputs]).mean()

		if split == 'train':
			self.log(f'train_loss_epoch', loss.item())

		else:
			preds   = torch.cat([torch.LongTensor(x['preds']) for x in outputs])
			targets = torch.cat([torch.LongTensor(x['targets']) for x in outputs])
			acc     = self.calc_acc(preds, targets)

			if split == 'valid':
				self.log(f'valid_loss_epoch', loss.item(), sync_dist=True)
				self.log(f'valid_acc_epoch', acc.item(), sync_dist=True)

			elif split == 'test':
				neg_indicator = torch.cat([x['neg_indicator'] for x in outputs]).bool()
				iid           = list(set([y for x in outputs for y in x['iid']]))
				theory_ids = [y for x in outputs for y in x['theory_ids']]
				assert len(iid) == 1
				prefix = 'iid' if iid[0] else 'ood'

				pred_neg, pred_noneg     = preds[neg_indicator], preds[~neg_indicator]
				target_neg, target_noneg = targets[neg_indicator], targets[~neg_indicator]

				# split by neg_indicator
				neg_acc   = 100 * (pred_neg == target_neg).float().mean()
				noneg_acc = 100 * (pred_noneg == target_noneg).float().mean()

				# split by label
				cf_mat_lbl = confusion_matrix(targets.tolist(), preds.tolist())

				# split by neg_indicator==1 and label
				cf_mat_neg_lbl = confusion_matrix(target_neg.tolist(), pred_neg.tolist())

				# split by neg_indicator==0 and label
				cf_mat_noneg_lbl = confusion_matrix(target_noneg.tolist(), pred_noneg.tolist())

				def get_cf_mat_accuracy(cf_mat, tgts):
					out_dict = dict()
					all_lbls = sorted(list(set(tgts.tolist())))
					for idx, lbl in enumerate(all_lbls):
						val                = np.round(cf_mat[idx][idx] / cf_mat.sum(1)[idx] * 100, 2)
						out_dict[f'{lbl}'] = val

					return out_dict

				lbl_acc       = get_cf_mat_accuracy(cf_mat_lbl, targets)
				neg_lbl_acc   = get_cf_mat_accuracy(cf_mat_neg_lbl, target_neg)
				noneg_lbl_acc = get_cf_mat_accuracy(cf_mat_noneg_lbl, target_noneg)

				# logging
				print('\n Label counts: ', cf_mat_lbl.sum(1))
				print('\n Neg and non-neg counts: ', sum(neg_indicator).item(), sum(~neg_indicator).item())
				print('\nLabel Confusion Matrix:')
				print(cf_mat_lbl)
				print('\nNeg Label Confusion Matrix:')
				print(cf_mat_neg_lbl)
				print('\nNon-Neg Label Confusion Matrix:')
				print(cf_mat_noneg_lbl)
				print('\nAcc, Label-wise Acc, Neg Acc, Non-neg Acc, Label-wise Neg Acc, Label-wise Non-Neg Acc')
				print(np.round(acc.item(), 2), lbl_acc, np.round(neg_acc.item(), 2), np.round(noneg_acc.item(), 2), neg_lbl_acc, noneg_lbl_acc)

				self.log(f'{prefix}_test_loss_epoch', loss.item(), sync_dist=True)
				self.log(f'{prefix}_test_acc_epoch', acc.item(), sync_dist=True)
				self.log(f'{prefix}_test_neg_acc', neg_acc.item(), sync_dist=True)
				self.log(f'{prefix}_test_noneg_acc', noneg_acc.item(), sync_dist=True)

				for k,v in lbl_acc.items():
					self.log(f'{prefix}_test_lbl_{k}_acc', v, sync_dist=True)

				for k,v in neg_lbl_acc.items():
					self.log(f'{prefix}_test_neg_lbl_{k}_acc', v, sync_dist=True)

				for k,v in noneg_lbl_acc.items():
					self.log(f'{prefix}_test_noneg_lbl_{k}_acc', v, sync_dist=True)

				outfile  = f'{self.p.save_dir}/{prefix}_output.csv'
				combined = list(zip(theory_ids, [x.item() for x in targets], [x.item() for x in preds]))
				with open(outfile, 'w') as f:
					writer = csv.writer(f)
					writer.writerows(combined)
				print('Written output at: ', outfile)
				if self.compute_metrics:
					outfile_eval = outfile.split("/")[-2]
					dict_eval = types.SimpleNamespace()
					dict_eval.outfile = outfile_eval
					dict_eval.grouped = True
					dict_eval.by_label = True
					dict_eval.grouped_and_label = False
					dict_eval.human_eval = False
					dict_eval.type = 'easy'
					print("This is easy metric")
					easy_res = eval_func.main(dict_eval)

					dict_eval.type = 'strict'
					print("This is strict metric")
					strict_res = eval_func.main(dict_eval)

					nl = '\n'
					print(f'Writing final results in {self.p.checkpoint_dir}/all_results.txt')
					with open(f'{self.p.checkpoint_dir}/all_results.txt', 'a') as f:
						dict_tags = ['Easy', 'Strict']
						for idx, res_dict in enumerate([easy_res, strict_res]):
							fcntl.flock(f, fcntl.LOCK_EX)
							f.write(f"{res_dict.pop('operator', None)} {dict_tags[idx]}: {self.p.save_dir}{nl}")
							f.write(f"overall{nl}{res_dict.pop('overall', None)}{nl}")
							f.write(f"label_0{nl}label_1{nl}label_2{nl}{res_dict.pop('label_0', None)},{res_dict.pop('label_1', None)},{res_dict.pop('label_2', None)}{nl}")
							keys, vals = [], []
							for k,v in res_dict.items():
								keys.append(k)
								vals.append(v)
							keys = '\n'.join(keys)
							vals = '\n'.join(vals)
							f.write(f"{keys}{nl}{vals}")
							f.write(f"{nl}{nl}")
						f.write(f'****************************{nl}{nl}')
						fcntl.flock(f, fcntl.LOCK_UN)


	def configure_optimizers(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params'      : [p for n, p in self.reasoner.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.reasoner.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		if self.p.optimizer == 'adamw':
			optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate)
		elif self.p.optimizer == 'adafactor':
			optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, clip_threshold=1.0, lr=self.p.learning_rate)
		else:
			raise NotImplementedError

		if self.p.lr_scheduler == 'linear_with_warmup':
			if self.p.warmup_updates > 1.0:
				warmup_steps = int(self.p.warmup_updates)
			else:
				warmup_steps = int(self.total_steps * self.p.warmup_updates)
			print(f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

			scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
			scheduler = {
				'scheduler': scheduler,
				'interval': 'step',
				'frequency': 1
			}
		elif self.p.lr_scheduler == 'fixed':
			return [optimizer]
		else:
			raise NotImplementedError

		return [optimizer], [scheduler]
