from transformers import ElectraTokenizer, AutoModel
import torch
import pandas as pd
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
import logging
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.classes_number)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NewsClf(nn.Module):

    def __init__(self, config):
        super(NewsClf, self).__init__()
        self.electra = AutoModel.from_pretrained(config.model, return_dict=True)
        self.classifier = ElectraClassificationHead(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.electra(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        output = pooled_output.last_hidden_state
        logits = self.classifier(output)
        return logits


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--save-dir", default='./models/news_clf_v1.1.pt', help="models files.")
parser.add_argument("--max-length", default=512)
parser.add_argument("--classes-number", default=38)
parser.add_argument("--dropout", default=0.2)
parser.add_argument("--model", default="hfl/chinese-electra-base-discriminator")
parser.add_argument("--hidden-size", default=768)  # 768 or 256
args = parser.parse_args()

# load label
df_label = pd.read_csv('./dict.csv', usecols=['label'])
label_list = df_label.label.tolist()
logging.info('label list {}'.format(label_list))

tokenizer = ElectraTokenizer.from_pretrained(args.model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

# initialize model
model = NewsClf(args)
model = model.to(device)
dict = torch.load(args.save_dir, map_location=device)
model.load_state_dict(dict['model_state_dict'])
print('best model epoch:', dict['epoch'])
model.eval()

print('model parameters: {}'.format(count_parameters(model)))


# class News():
#     def __init__(self, s):
#         self.text = (s.unsqueeze(1), torch.tensor([len(s)]).to(device))

def predict_label(t, c):
    start_time = time.time()
    encoding = tokenizer.encode_plus(
        t,
        c,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=args.max_length,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        preds = output.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    print('lost time: {}, {}'.format(mins, secs))
    return preds[0]

predict_label('搞笑gif图：这就是周末的我！能瘫着就不起来！', '这就是周末的我！能瘫着就不起来！    佛系的熊    看不惯成双成对    嘴：我不怕 脑袋：我不想死    呵，男人    哇，真好吃 我要继续吃    驴子：咱不带这样玩的！    应该是一个教练带出来的吧    因为主人追剧而被忽视的喵，正在墙角生闷气......')