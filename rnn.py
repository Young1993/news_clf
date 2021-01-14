import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchtext import data
from torchtext import vocab
import jieba
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from sklearn import metrics
from tensorboardX import SummaryWriter
import time
import logging


def classifiction_metric(preds, labels, label_list):
    acc = metrics.accuracy_score(labels, preds)
    labels_list = [i for i in range(len(label_list))]
    report = metrics.classification_report(
        labels, preds, labels=labels_list, target_names=label_list, digits=4, output_dict=True)
    return acc, report


class Config(object):
    """配置参数"""

    def __init__(self, embedding, type):
        self.clip = 10
        self.model_name = 'rnn'
        self.class_list = ['教育', '财经', '时政', '科技', '社会', '健康', '其他']
        # self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = './models/' + self.model_name + '.pt'  # 模型训练结果
        self.log_path = './logs/' + self.model_name
        # embedding size
        self.embedding_file = embedding
        self.embed = 300
        # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = 0.3  # 随机失活
        self.require_improvement = 1000  # 若超过 1000 batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 60  # epoch数
        self.bidirectional = True
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-2  # 学习率
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数

        if type == 'test':
            self.batch_size = 16  # mini-batch 大小 128
            self.n_vocab = 1000  # 词表大小，在运行时赋值
            self.data_path = './data/test/'
            self.print_step = 2  # 100
        else:
            self.batch_size = 128  # mini-batch 大小 128
            self.n_vocab = 15000  # 15000 词表大小，在运行时赋值
            self.data_path = './fold/'  # './fold/'
            self.print_step = 100  # 100


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class News_clf(nn.Module):
    def __init__(self, config, word_emb):
        super(News_clf, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_emb)

        self.lstm = nn.LSTM(input_size=config.embed, hidden_size=config.hidden_size, num_layers=config.num_layers,
                            bidirectional=config.bidirectional, batch_first=True, dropout=config.dropout)

        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.dropout = nn.Dropout(config.dropout)

        if config.bidirectional:
            input_features = config.hidden_size * 2
        else:
            input_features = config.hidden_size

        self.fc = nn.Linear(input_features, config.num_classes)

    def attention_net(self, x, query, mask=None):  # 软性注意力机制（key=value=x）
        dv = query.size(1)
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1) / dv  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x):
        text, _ = x
        # title_emb: [seq_len, batch_size, emd_dim]
        text_emb = self.dropout(self.word_embedding(text))

        # [batch_size, emd_dim, seq_len]
        text_emb = text_emb.permute(1, 0, 2)

        H, _ = self.lstm(text_emb)  # [batch_size, seq_len, hidden_size * num_direction]

        query = self.dropout(H)

        attn_output, attention = self.attention_net(H, query)

        out = self.fc(attn_output)

        return out


def load_embedding(file):
    f = open(file, "r", encoding='UTF-8')
    embeddings = {}
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        emb = [float(x) for x in lin[1:301]]
        embeddings[lin[0]] = np.asarray(emb, dtype='float32')
    f.close()
    return embeddings


# load data
def load_news(config, text_field, band_field):
    fields = {
        'text': ('text', text_field),
        'label': ('label', band_field)
    }

    word_vectors = vocab.Vectors(config.embedding_file)

    train, val, test = data.TabularDataset.splits(
        path=config.data_path, train='train.csv', validation='val.csv',
        test='test.csv', format='csv', fields=fields)

    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(val.examples), len(test.examples)))

    text_field.build_vocab(train, val, test, max_size=config.n_vocab, vectors=word_vectors,
                           unk_init=torch.Tensor.normal_)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(config.batch_size, config.batch_size, config.batch_size), sort=False,
        device=config.device, sort_within_batch=False, shuffle=False)

    return train_iter, val_iter, test_iter


# start loggging
logging.basicConfig(filemode='w', filename="./logs/log.txt", level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
embedding = './data/sgns.sogou.word'

type = 'train'

config = Config(embedding, type)
logging.info('config:{}'.format(config))
logging.info('device:{}'.format(config.device))


def tokenizer(s):
    return jieba.lcut(s)


# data loader and split Chinese
text_field = data.Field(tokenize=tokenizer, include_lengths=True, fix_length=config.pad_size)

band_field = data.Field(sequential=False, use_vocab=False, batch_first=True,
                        dtype=torch.int64, preprocessing=data.Pipeline(lambda x: int(x)))

train_iterator, val_iterator, test_iterator = load_news(config, text_field, band_field)


# initialize
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


word_emb = text_field.vocab.vectors
model = News_clf(config, word_emb)

logging.info('model parameters: {}'.format(count_parameters(model)))
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


def poly_scheduler(epoch, num_epochs=config.num_epochs, power=0.9):
    return (1 - epoch / num_epochs) ** power


scheduler = LambdaLR(optimizer, lr_lambda=poly_scheduler)

criterion = nn.CrossEntropyLoss()
model = model.to(config.device)
criterion = criterion.to(config.device)


# 评估模型
def evaluate(model, iterator, criterion, config):
    model.eval()
    epoch_loss = 0

    # all_preds = np.array([], dtype=int)
    # all_labels = np.array([], dtype=int)

    with torch.no_grad():
        for batch in iterator:
            logits = model(batch.text)
            loss = criterion(logits.view(-1, config.num_classes), batch.label)
            epoch_loss += loss.item()

            # label = batch.label.detach().cpu().numpy()
            # preds = logits.detach().cpu().numpy()
            # preds = np.argmax(preds, axis=1)
            # all_preds = np.append(all_preds, preds)
            # all_labels = np.append(all_labels, label)

    # acc, report = classifiction_metric(all_preds, all_labels, config.class_list)

    return epoch_loss / len(iterator)  # , acc, report


def train(model, train_iterator, val_iterator, optimizer, criterion, config):
    model.train()

    best_loss = float('inf')
    global_step = 0

    writer = SummaryWriter(
        log_dir=config.log_path + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))

    for epoch in range(config.num_epochs):
        print('---------------- Epoch: ' + str(epoch) + ' + 1:02 ----------')
        logging.info('---------------- Epoch: ' + str(epoch) + ' ----------')
        epoch_loss = 0
        train_steps = 0
        # all_preds = np.array([], dtype=int)
        # all_labels = np.array([], dtype=int)

        for step, batch in enumerate(train_iterator):
            optimizer.zero_grad()

            logits = model(batch.text)
            loss = criterion(logits.view(-1, config.num_classes), batch.label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            train_steps += 1

            # label = batch.label.detach().cpu().numpy()
            # preds = logits.detach().cpu().numpy()
            #
            # preds = np.argmax(preds, axis=1)
            # all_preds = np.append(all_preds, preds)
            # all_labels = np.append(all_labels, label)

            if global_step % config.print_step == 0:
                train_loss = epoch_loss / train_steps
                # 指标比较
                # train_acc, train_report = classifiction_metric(all_preds, all_labels, config.class_list)
                # val_loss, val_acc, val_report = evaluate(model, val_iterator, criterion, config)

                val_loss = evaluate(model, val_iterator, criterion, config)

                c = global_step // config.print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/val", val_loss, c)

                # writer.add_scalar("acc/train", train_acc, c)
                # writer.add_scalar("acc/val", val_acc, c)

                logging.info("loss/train: %0.3f, %d" % (train_loss, c))
                logging.info("loss/val: %0.3f, %d" % (val_loss, c))
                # logging.info("acc/train: %0.3f, %d" % (train_acc, c))
                # logging.info("acc/val: %0.3f, %d" % (val_acc, c))

                # for label in config.class_list:
                #     writer.add_scalar(label + ":f1/train", train_report[label]['f1-score'], c)
                #     writer.add_scalar(label + ":f1/dev", val_report[label]['f1-score'], c)
                #     print(label + ":f1/train", train_report[label]['f1-score'], c)
                #     print(label + ":f1/dev", val_report[label]['f1-score'], c)

                # writer.add_scalar("weighted avg:f1/train", train_report['weighted avg']['f1-score'], c)
                # writer.add_scalar("weighted avg:f1/dev", val_report['weighted avg']['f1-score'], c)

                # logging.info("weighted avg:f1/train: %0.3f, %d " % (train_report['weighted avg']['f1-score'], c))
                # logging.info("weighted avg:f1/dev %0.3f, %d " % (val_report['weighted avg']['f1-score'], c))

                if best_loss > val_loss:
                    best_loss = val_loss
                    logging.info('=' * 50)
                    logging.info('best_loss: %0.3f ' % best_loss)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict()
                                },
                               config.save_path)

                model.train()
        scheduler.step()
        logging.info('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))


# start training
train(model, train_iterator, val_iterator, optimizer, criterion, config)

logging.info('finished training')

def evaluation(model, iterator, config):
    model.eval()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    with torch.no_grad():
        for batch in iterator:
            logits = model(batch.text)
            label = batch.label.detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, label)
    return classifiction_metric(all_preds, all_labels, config.class_list)

# evaluation
logging.info("-------------- Test -------------")

test_loss, test_acc, test_report = evaluation(model, test_iterator, config)
logging.info('-' * 50)
logging.info("\t test Loss: {}, \t test acc: {}, \t test report: {}".format(test_loss, test_acc, test_report))
