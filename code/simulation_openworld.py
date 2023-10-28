"""
File: simulation_openworld.py
Author: lok
Edited By: jzx-bupt
"""
from models import *
from metrics import *
import pickle

cls_num = 51
max_page = 6
page_len = 5120
batch_size = 80
lr = 0.0005

tab2path = {}
tab2path[2] = r'D:\论文工作\多页指纹攻击\datasets\merge with duration\tbb_2tabs_er5'
tab2path[4] = r'D:\论文工作\多页指纹攻击\datasets\merge with duration\tbb_4tabs_er5'
tab2path[6] = r'D:\论文工作\多页指纹攻击\datasets\merge with duration\tbb_6tabs_er5'

# just for tunning
nhead = 8
num_encoder_layers = 2
num_decoder_layers = 2


def load_data_Ntab(tab, pages=6, timestamp=False, class_total=51):
    train_ret = {}
    test_ret = {}

    train_ret['data'] = []
    train_ret['label'] = []
    test_ret['data'] = []
    test_ret['label'] = []

    if timestamp:
        train_ret['time'] = []
        test_ret['time'] = []

    for i in range(3):
        if tab != 0:
            f = open(tab2path[tab] + '/COCO_train_' + str(i), 'rb')
        else:
            f = open(tab2path[6] + '/COCO_train_' + str(i), 'rb')
        raw_data = pickle.load(f)
        train_ret['data'].extend(raw_data['data'])
        if timestamp:
            train_ret['time'].extend(raw_data['time'])
        f.close()

    if tab != 0:
        train_ret['label'] = np.concatenate([np.load(tab2path[tab] + '/BAPM_train.npz')['label'],
                                             np.ones((len(train_ret['data']), pages - tab)) * (class_total - 1)],
                                            axis=-1).tolist()
    else:
        train_ret['label'] = np.concatenate([np.load(tab2path[6] + '/BAPM_train.npz')['label'],
                                             np.ones((len(train_ret['data']), pages - 6)) * (class_total - 1)],
                                            axis=-1).tolist()

    for pt in [2, 4, 6]:
        if pt == tab or tab == 0:
            f = open(tab2path[pt] + '/COCO_test', 'rb')
            raw_data = pickle.load(f)
            test_ret['data'].extend(raw_data['data'])
            if timestamp:
                test_ret['time'].extend(raw_data['time'])
            f.close()

            test_ret['label'].extend(np.concatenate([np.load(tab2path[pt] + '/BAPM_test.npz')['label'],
                                                     np.ones((len(test_ret['data']), pages - pt)) * (class_total - 1)],
                                                    axis=-1).tolist())

    return train_ret, test_ret


class DataGenerator(object):
    def __init__(self, batch_size=batch_size, dim=page_len, page=max_page, timestamp=False):
        # Initialization
        self.batch_size = batch_size
        self.dim = dim
        self.page = page
        self.timestamp = timestamp

    def generate(self, data, indices):
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
                    tlen = len(data['data'][k])
                    x_dire = data['data'][k]
                    if self.timestamp:
                        x_time = data['time'][k]
                        max_time = x_time[np.flatnonzero(x_time)[-1]]
                        for l, t in enumerate(x_time):
                            x_time[l] = t / max_time
                        x_time = np.array(x_time)
                        if tlen >= max_len:
                            x.append([x_dire.tolist()[:max_len], x_time.tolist()[:max_len]])
                        else:
                            x.append(
                                [x_dire.tolist() + [0] * (max_len - tlen), x_time.tolist() + [0] * (max_len - tlen)])
                    else:
                        if tlen >= max_len:
                            x.append(x_dire.tolist()[:max_len])
                        else:
                            x.append(x_dire.tolist() + [0] * (max_len - tlen))
                    y.append(data['label'][k])
                yield np.array(x), np.array(y)


def train_test(backbone='DFNet', tab=2, page_dim=page_len, max_page=max_page, timestamp=False):
    np.random.seed(2023)
    torch.manual_seed(2023)

    train_ret, test_ret = load_data_Ntab(tab, max_page, timestamp, cls_num)
    train_total = len(train_ret['data'])
    indices = np.arange(train_total)
    np.random.shuffle(indices)

    train_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(train_ret, indices)

    test_total = len(test_ret['data'])
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(test_ret, indices)

    if backbone == 'BAPM-CNN':
        model = TMWF_noDF(embed_dim=128, nhead=8, dim_feedforward=512, num_encoder_layers=2,
                          num_decoder_layers=2, max_len=page_dim * max_page // (8 * 8 * 8), num_queries=max_page,
                          cls=cls_num, dropout=0.1).cuda()
    elif backbone == 'DFNet':
        model = TMWF_DFNet(embed_dim=256, nhead=8, dim_feedforward=256 * 4, num_encoder_layers=2,
                           num_decoder_layers=2, max_len=121, num_queries=max_page, cls=cls_num,
                           dropout=0.1).cuda()

    # state_dict = torch.load(r'D:\论文工作\多页指纹攻击\TMWF\model\ck_TDN_6tab_.pth')['state_dict']
    # model.load_state_dict(state_dict)

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


def parameter_tunning(backbone='BAPM-CNN', tab=2, page_dim=page_len, max_page=max_page, timestamp=False):
    np.random.seed(2023)
    torch.manual_seed(2023)
    train_ret, test_ret = load_data_Ntab(tab, max_page, timestamp, cls_num)

    train_total = len(train_ret['data'])
    indices = np.arange(train_total)
    np.random.shuffle(indices)
    train_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(train_ret, indices)

    test_total = len(test_ret['data'])
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(test_ret, indices)

    if backbone == 'BAPM-CNN':
        model = TMWF_noDF(embed_dim=128, nhead=nhead, dim_feedforward=512, num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers, max_len=page_dim * max_page // (8 * 8 * 8), num_queries=max_page,
                          cls=cls_num, dropout=0.1).cuda()
    elif backbone == 'DFNet':
        model = TMWF_DFNet(embed_dim=256, nhead=nhead, dim_feedforward=256 * 4, num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers, max_len=121, num_queries=max_page, cls=cls_num,
                           dropout=0.1).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    criteron = torch.nn.CrossEntropyLoss()

    print('nhead:', nhead)
    print('num_encoder_layers:', num_encoder_layers)
    print('num_decoder_layers:', num_decoder_layers)

    best_score = 0.
    stop_count = 0

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
        acc1 = overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
        pre1 = overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
        rec1 = overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
        acc2 = overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
        pre2 = overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
        rec2 = overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)

        score = (acc1 + acc2 + pre1 + pre2 + rec1 + rec2) / 6

        if score >= best_score:
            best_score = score
            stop_count = 0
            torch.save({'state_dict': model.state_dict()},
                       'model/pt_{}_h{}e{}d{}_{}tab_.pth'.format(backbone, nhead, num_encoder_layers,
                                                                 num_decoder_layers, tab))
        else:
            stop_count = stop_count + 1
        print('best_score: ', best_score)


def evaluate(model_path, dim=page_len, max_page=max_page, tab=0):
    # timestamp = True
    timestamp = False

    _, test_ret = load_data_Ntab(tab, max_page, timestamp, cls_num)
    test_total = len(test_ret['data'])
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, dim, max_page, timestamp).generate(test_ret, indices)

    model = TMWF_DFNet(embed_dim=256, nhead=8, dim_feedforward=256 * 4, num_encoder_layers=2,
                       num_decoder_layers=2, max_len=121, num_queries=max_page, cls=cls_num, dropout=0.1).cuda()

    state_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(state_dict)

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

    previous_accuracy(prob_matrix, one_hot_matrix, max_page)
    previous_precision(prob_matrix, one_hot_matrix, cls_num, max_page)
    previous_recall(prob_matrix, one_hot_matrix, cls_num, max_page)

    overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)


def train_only(tab=2, page_dim=page_len, page_total=max_page, timestamp=False):
    np.random.seed(2023)
    torch.manual_seed(2023)

    train_ret, test_ret = load_data_Ntab(tab, page_total, timestamp, cls_num)

    train_total = len(train_ret['data'])
    indices = np.arange(train_total)
    np.random.shuffle(indices)
    train_gen = DataGenerator(batch_size, page_dim, page_total, timestamp).generate(train_ret, indices)

    test_total = len(test_ret['data'])
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, page_dim, page_total, timestamp).generate(test_ret, indices)

    model = TMWF_DFNet(embed_dim=256, nhead=4, dim_feedforward=256 * 4, num_encoder_layers=1,
                       num_decoder_layers=1, max_len=121, num_queries=page_total, cls=cls_num,
                       dropout=0.1).cuda()

    from torchinfo import summary
    summary(model)
    return

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
            for ct in range(page_total):
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

    overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
    overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)


if __name__ == '__main__':
    import time
    start = time.time()

    # train_test(backbone='DFNet', tab=6, page_dim=page_len, page_total=max_page, timestamp=True)
    evaluate(model_path=r'D:\论文工作\多页指纹攻击\TMWF\model\merge with duration\racknerd_tbb_6tabs_.pth', dim=page_len,
             max_page=max_page, tab=2)
    # train_only(tab=6, page_dim=page_len, max_page=max_page, timestamp=False)
    # parameter_tunning(backbone='DFNet', tab=6, page_dim=page_len, max_page=max_page, timestamp=False)

    end = time.time()
    print(end - start, 's')
