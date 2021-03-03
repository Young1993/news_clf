from transformers import ElectraTokenizer, AutoModel

import torch
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from sklearn import metrics
import time


def classifiction_metric(preds, labels, label_list):
    acc = metrics.accuracy_score(labels, preds)
    labels_list = [i for i in range(len(label_list))]
    report = metrics.classification_report(
        labels, preds, labels=labels_list, target_names=label_list, digits=4, output_dict=True)
    return acc, report


def statistics(tokenizer, df_train):
    import seaborn as sns
    import matplotlib.pyplot as plt

    token_lens = []

    for txt in df_train['content']:
        tokens = tokenizer(txt)
        token_lens.append(len(tokens['input_ids']))

    sns.distplot(token_lens)
    plt.xlim([0, 300])
    plt.xlabel('Token count')
    plt.show()


class TQReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len, config):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': target
        }


def create_data_loader(df, tokenizer, args):
    ds = TQReviewDataset(
        reviews=df.text.to_numpy(),  # text
        targets=df.label.to_numpy(),  # single labels
        tokenizer=tokenizer,
        max_len=args.max_length,
        config=args
    )

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=0
    )


# 评估模型
def evaluate(model, iterator, device, criterion, config):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["targets"].to(device)

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = criterion(output.view(-1, config.classes_number), labels)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


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


# fine tune
def run_fine_tune(args):
    # load data
    df_train = pd.read_csv(args.data_dir + '/train.csv')
    df_val = pd.read_csv(args.data_dir + '/val.csv')
    df_test = pd.read_csv(args.data_dir + '/test.csv')

    logging.info('len of df_train: %d' % len(df_train))
    logging.info('len of df_val: %d' % len(df_val))
    logging.info('len of df_test: %d' % len(df_test))

    # load label
    df_label = pd.read_csv('./dict.csv', usecols=['label'])
    label_list = df_label.label.tolist()
    logging.info('label list {}'.format(label_list))

    tokenizer = ElectraTokenizer.from_pretrained(args.model)

    # set random seed
    random_seed = 1432
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device: %s' % device)
    logging.info("device: %s" % device)

    logging.info("loader the data and processed...")
    train_data_loader = create_data_loader(df_train, tokenizer, args)
    val_data_loader = create_data_loader(df_val, tokenizer, args)
    test_data_loader = create_data_loader(df_test, tokenizer, args)

    # initialize model
    model = NewsClf(args)
    model = model.to(device)

    logging.info('model parameters: {}'.format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    def poly_scheduler(epoch, num_epochs=args.epoch, power=0.95):
        return (1 - epoch / num_epochs) ** power

    scheduler = LambdaLR(optimizer, lr_lambda=poly_scheduler)

    global_steps = 0
    best_loss = float('inf')

    writer = SummaryWriter(
        log_dir=args.logs + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))

    # start training
    logging.info('start training ...')
    for epoch in range(args.epoch):
        start_time = time.time()
        logging.info('(' + str(epoch) + '/' + str(args.epoch) + ')')

        model = model.train()
        epoch_loss = 0
        train_steps = 0

        for d in train_data_loader:
            optimizer.zero_grad()

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            labels = d["targets"].to(device)

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = criterion(output.view(-1, args.classes_number), labels)
            epoch_loss += loss.item()
            loss.backward()

            train_steps += 1
            global_steps += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if global_steps % args.print_step == 0:
                train_loss = epoch_loss / train_steps
                val_loss = evaluate(model, val_data_loader, device, criterion, args)

                c = global_steps // args.print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/val", val_loss, c)

                logging.info("loss/train: %0.3f, %d" % (train_loss, c))
                logging.info("loss/val: %0.3f, %d" % (val_loss, c))

                if val_loss < best_loss:
                    best_loss = val_loss
                    logging.info("-" * 50)
                    logging.info('|| best loss: %0.3f ' % best_loss)

                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'loss': best_loss}, args.save_dir)
                model.train()
        logging.warning('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        logging.info("Epoch: {} | time in {} minutes, {} seconds".format(epoch + 1, mins, secs))

    logging.info('finished train')
    test_loss = evaluate(model, test_data_loader, device)
    logging.info("-" * 50)
    logging.info("|| test loss: %0.3f " % test_loss)


# evaluate model
def run_evaluate(args):
    df_test = pd.read_csv(args.data_dir + '/test.csv')
    df_label = pd.read_csv('./dict.csv', usecols=['label'])
    labels = df_label.label.tolist()
    logging.info('label list {}'.format(labels))

    tokenizer = ElectraTokenizer.from_pretrained(args.model)
    test_data_loader = create_data_loader(df_test, tokenizer, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NewsClf(args)
    model = model.to(device)

    dict = torch.load(args.save_dir, map_location=device)
    model.load_state_dict(dict['model_state_dict'])
    print('best model epoch:', dict['epoch'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["targets"].to(device)

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            label = labels.detach().cpu().numpy()
            preds = output.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, label)

    return classifiction_metric(all_preds, all_labels, args.classes_number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default='./data', help="新闻数据文件")
    parser.add_argument("--logs", default='./logs/', help="Location of logs files.")
    parser.add_argument("--save-dir", default='./models/news_clf_v1.0.pt', help="models files.")
    parser.add_argument("--max-length", default=512)
    parser.add_argument("--batch-size", default=16)  # 256 or 128
    parser.add_argument("--epoch", default=15)
    parser.add_argument("--clip", default=10)
    parser.add_argument("--learning-rate", default=3e-4)
    parser.add_argument("--print-step", default=100)  # 100
    parser.add_argument("--classes-number", default=38)

    parser.add_argument("--dropout", default=0.2)
    parser.add_argument("--model",
                        default="hfl/chinese-electra-base-discriminator")  # hfl/chinese-electra-base-discriminator
    parser.add_argument("--hidden-size", default=768)  # 768 or 256
    parser.add_argument("--train", default=True)

    args = parser.parse_args()
    logging.basicConfig(filemode='w', filename="./logs/train.txt", level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.train:
        logging.info('training...')
        run_fine_tune(args)
    else:
        logging.info('evaluate...')
        run_evaluate(args)


if __name__ == '__main__':
    main()
