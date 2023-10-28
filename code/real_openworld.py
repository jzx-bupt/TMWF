"""
File: real_openworld.py
Author: lok
Edited By: jzx-bupt
"""
import os
from models import *
from metrics import *
import pickle

cls_num = 51
max_page = 6
page_len = 5120
batch_size = 80
lr = 0.0005


def load_data_individual(tab_number):
    x_train, y_train = [], []
    # Load training data from multiple files
    for i in range(3):
        file_path = f'dataset/racknerd_{tab_number}tabs_er5/COCO_train_{i}'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)['data']
            x_train.extend(data)

        if i == 0:
            y_train_temp = np.load(f'dataset/racknerd_{tab_number}tabs_er5/BAPM_train.npz')['label']
            y_train_temp = y_train_temp.tolist()
            y_train.extend(y_train_temp)
        else:
            y_train.extend([50] * len(data))

    # Load test data
    with open('dataset/racknerd_{}tabs_er5/COCO_test'.format(tab_number), 'rb') as f:
        x_test = pickle.load(f)['data']

    y_test_temp = np.load(f'dataset/racknerd_{tab_number}tabs_er5/BAPM_test.npz')['label']
    y_test_temp = y_test_temp.tolist()
    y_test = y_test_temp + [50] * len(x_test)

    return x_train, y_train, x_test, y_test


def load_data_train_and_val(num_tabs, page=6, dir_path='../datasets', dataset='tbb'):
    train_files = [f'{dir_path}/{dataset}_{num_tabs}tabs_er10/COCO_train_{i}' for i in range(4)]
    x_train = [pickle.load(open(f, 'rb'))['data'] for f in train_files]
    x_train = [x for l in x_train for x in l]  # flatten the list
    y_train_temp = np.load(f'{dir_path}/{dataset}_{num_tabs}tabs_er10/BAPM_train.npz')['label']
    y_train = np.concatenate([y_train_temp, np.ones((len(x_train), page - num_tabs)) * 50], axis=-1).tolist()

    valid_file = f'{dir_path}/{dataset}_{num_tabs}tabs_er10/COCO_test'
    x_val = pickle.load(open(valid_file, 'rb'))['data']
    y_val_temp = np.load(f'{dir_path}/{dataset}_{num_tabs}tabs_er10/BAPM_test.npz')['label']
    y_val = np.concatenate([y_val_temp, np.ones((len(x_val), page - num_tabs)) * 50], axis=-1).tolist()

    return x_train, y_train, x_val, y_val


def load_real_world(tab, max_page, dim, front_part=1000, dataset='chrome'):
    x_data = []
    y_data = []

    tab_list = [2, 3, 4, 5] if dataset == 'chrome' else [2, 3, 4]

    for t in tab_list:
        if tab != t:
            filepath = os.path.join('D:/论文工作/多页指纹攻击/datasets', f'{dataset} realworld', f'{t}tabs')
            with open(filepath, 'rb') as f:
                raw_data = pickle.load(f)
            data = raw_data['data'][:front_part]
            total = len(data)
            y_data.append(np.concatenate([np.array(raw_data['label'][:front_part]), np.ones((total, max_page - t)) * 50], axis=-1))
            x_dir = []
            max_len = max_page * dim
            for idx in range(total):
                lt = len(data[idx])
                x_dir.append(data[idx].tolist()[:max_len] + [0] * (max_len - lt))
            x_data.append(np.array(x_dir))

    if tab in tab_list:
        return x_data[tab_list.index(tab)], y_data[tab_list.index(tab)]
    else:
        return np.concatenate(x_data, axis=0), np.concatenate(y_data, axis=0)


class DataGenerator(object):
    def __init__(self, batch_size=batch_size, dim=page_len, page=max_page, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.page = page

    def generate(self, data, labels, indices):
        max_len = self.dim * self.page
        total = len(indices)
        imax = int(len(indices) / self.batch_size)
        if total % self.batch_size != 0:
            imax = imax + 1
        while True:
            for i in range(imax):
                x = []
                y = []
                for j, k in enumerate(indices[i * self.batch_size:(i + 1) * self.batch_size]):
                    tlen = len(data[k])
                    if tlen >= max_len:
                        x.append(data[k].tolist()[:max_len])
                    else:
                        x.append(data[k].tolist() + [0] * (max_len - tlen))
                    y.append(labels[k])
                yield np.array(x), np.array(y)


def train_val(page_dim=5120, max_page=6):
    np.random.seed(2023)
    torch.manual_seed(2023)
    cls_num = 51
    x_train, y_train, x_test, y_test = load_data_train_and_val(max_page, dir_path=r'D:\论文工作\多页指纹攻击\datasets',
                                                               dataset='tbb')
    # load_data_individual(max_page)

    train_total = len(x_train)
    indices = np.arange(train_total)
    np.random.shuffle(indices)

    train_gen = DataGenerator(batch_size, page_dim, max_page).generate(x_train, y_train, indices)
    test_total = len(x_test)
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, page_dim, max_page).generate(x_test, y_test, indices)
    model = TMWF_DFNet(embed_dim=256, nhead=8, dim_feedforward=256 * 4, num_encoder_layers=2,
                       num_decoder_layers=2, max_len=121, num_queries=max_page, cls=cls_num, dropout=0.1).cuda()
    # model.load_state_dict(torch.load('model/TWF_6tab_0.pth')['state_dict'])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criteron = torch.nn.CrossEntropyLoss()

    train_bc = train_total // batch_size
    if train_bc * batch_size != train_total:
        train_bc = train_bc + 1
    test_bc = test_total // batch_size
    if test_bc * batch_size != test_total:
        test_bc = test_bc + 1

    for epoch in range(50):
        model.train()
        count = 0
        for xb, yb in train_gen:
            log = model(torch.tensor(xb, dtype=torch.float).cuda())
            y_batch = torch.tensor(yb, dtype=torch.long).cuda()
            opt.zero_grad()
            loss = 0
            for ct in range(max_page):
                loss_ct = criteron(log[:, ct], y_batch[:, ct].cuda())
                loss = loss + loss_ct
            loss.backward()
            opt.step()
            count = count + 1
            if count == train_bc:
                break

        model.eval()
        probs = []
        one_hot = []
        count = 0

        for xb, yb in test_gen:
            log = model(torch.tensor(xb, dtype=torch.float).cuda())
            y_batch = torch.tensor(yb, dtype=torch.long).cuda()
            probs.append(log.data.cpu().numpy())
            one_hot.append(F.one_hot(y_batch, cls_num).data.cpu().numpy())
            count = count + 1
            if count == test_bc:
                break

        prob_matrix = np.concatenate(probs, axis=0)
        one_hot_matrix = np.concatenate(one_hot, axis=0)

        print('Epoch', epoch)
        overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
        overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
        overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
        overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
        overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
        overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)


def evaluate_real(model_path, dim, page_total, tab=0, front_part=1000, dataset='chrome'):
    x_test, y_test = load_real_world(tab, page_total, dim, front_part, dataset)

    model = TMWF_DFNet(embed_dim=256, nhead=8, dim_feedforward=256 * 4, num_encoder_layers=2,
                       num_decoder_layers=2, max_len=121, num_queries=page_total, cls=cls_num, dropout=0.1).cuda()

    state_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(state_dict)

    test_total = len(x_test)
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, dim, page_total).generate(x_test, y_test, indices)
    test_bc = test_total // batch_size
    if test_bc * batch_size != test_total:
        test_bc = test_bc + 1
    model.eval()
    probs = []
    one_hot = []
    count = 0
    for xb, yb in test_gen:
        log = model(torch.tensor(xb, dtype=torch.float).cuda())
        y_batch = torch.tensor(yb, dtype=torch.long).cuda()
        probs.append(log.data.cpu().numpy())
        one_hot.append(F.one_hot(y_batch, cls_num).data.cpu().numpy())
        count = count + 1
        if count == test_bc:
            break
    prob_matrix = np.concatenate(probs, axis=0)
    one_hot_matrix = np.concatenate(one_hot, axis=0)

    previous_accuracy(prob_matrix, one_hot_matrix, page_total)
    previous_precision(prob_matrix, one_hot_matrix, cls_num, page_total)
    previous_recall(prob_matrix, one_hot_matrix, cls_num, page_total)

    overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)



if __name__ == '__main__':
    # train_val(page_len, max_page)
    evaluate_real(model_path=r'model\chrome_6tabs_duration_40k.pth', dim=page_len, page_total=max_page, tab=0)
    # model = TWF(embed_dim=128,nhead=4,dim_feedforward=512,num_encoder_layers=1,
    #         num_decoder_layers=1,max_len=5120*6//(8*8*8),num_queries=6,cls=51,dropout=0.1).cuda()
