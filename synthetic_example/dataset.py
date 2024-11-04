import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataloaders(X_train, X_test, y_train, y_test, batch_size, seed=None):
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True,
                              shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=False,
                             shuffle=False, generator=g)
    return train_loader, test_loader


class Sampler(object):
    @staticmethod
    def sample_circles(y, multiview_mask, margin):
        num_samples = y.shape[0]

        # apply inverse transform for sampling radius
        u = torch.rand(num_samples, device=y.device)
        r = torch.where(
            y == 0,
            np.sqrt(u * (1 - margin) ** 2),
            np.sqrt((1 + margin) ** 2 + u * (4 - (1 + margin) ** 2))
        )
        phi = 2 * torch.pi * torch.rand(num_samples, device=y.device)

        x1 = torch.where(multiview_mask, r * torch.cos(phi), torch.cos(phi))
        x2 = torch.where(multiview_mask, r * torch.sin(phi), torch.sin(phi))
        return x1, x2

    @staticmethod
    def sample_sectors(y, multiview_mask, margin):
        num_samples = y.shape[0]
        rand_sign = (2 * (torch.rand(num_samples, device=y.device) > 0.5).to(torch.long) - 1)
        x1 = (margin + 2 * torch.rand(num_samples, device=y.device)) * rand_sign
        x2 = margin + 2 * torch.rand(num_samples, device=y.device)
        x2 = torch.where(y == 0, x2 * rand_sign, -x2 * rand_sign)

        border_mask = (torch.rand(num_samples, device=y.device) > 0.5).to(torch.float)
        x1_border = -2 - margin + (4 + 2 * margin) * torch.rand(num_samples, device=y.device)
        x1_border = x1_border * border_mask
        x2_border = -2 - margin + (4 + 2 * margin) * torch.rand(num_samples, device=y.device)
        x2_border = x2_border * (1 - border_mask)

        x1 = torch.where(multiview_mask, x1, x1_border)
        x2 = torch.where(multiview_mask, x2, x2_border)
        return x1, x2

    @staticmethod
    def sample_stripe(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()

        r = 4 - 2 * np.sqrt(2)
        r_low = r - margin * np.sqrt(2)
        r_high = r + margin * np.sqrt(2)

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)
            neg_mask = (x[:, 1] <= x[:, 0] + r_low) & (x[:, 1] >= x[:, 0] - r_low)
            pos_mask = (x[:, 1] > x[:, 0] + r_high) | (x[:, 1] < x[:, 0] - r_high)
            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        rand_sign = (2 * (torch.rand(num_samples, device=y.device) > 0.5).to(torch.long) - 1)
        x1_border = (-2 + (4 - r) * torch.rand(num_samples, device=y.device)) * rand_sign
        x2_border = x1_border + r * rand_sign

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]

    @staticmethod
    def sample_diamond(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()

        r = 2
        r_low = r - margin * np.sqrt(2)
        r_high = r + margin * np.sqrt(2)

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)
            neg_mask = torch.abs(x).sum(1) <= r_low
            pos_mask = torch.abs(x).sum(1) > r_high
            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        rand_sign = (2 * (torch.rand(num_samples, device=y.device) > 0.5).to(torch.long) - 1)
        x1_border = -2 + 4 * torch.rand(num_samples, device=y.device)
        x2_border = rand_sign * (r - torch.abs(x1_border))

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]

    @staticmethod
    def sample_peaks(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()
        m = margin * np.sqrt(10)

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)
            r1, r2 = -3 * x[:, 0] - 4, 3 * x[:, 0] + 2
            r3, r4 = 3 * x[:, 0] - 2, -3 * x[:, 0] + 4
            neg_mask = ((x[:, 1] >= r1 + m) & (x[:, 1] >= r2 + m)) | \
                       ((x[:, 1] <= r3 - m) & (x[:, 1] <= r4 - m))
            pos_mask = (
                ((x[:, 1] <= r1 - m) | (x[:, 1] <= r2 - m)) & (x[:, 1] >= 3 * x[:, 0])
            ) | (
                ((x[:, 1] >= r3 + m) | (x[:, 1] >= r4 + m)) & (x[:, 1] < 3 * x[:, 0])
            )
            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        x1_border = -2 + 4 * torch.rand(num_samples, device=y.device)
        x2_border = torch.zeros_like(x1_border)

        m = x1_border < -1
        x2_border[m] = -3 * x1_border[m] - 4
        m = (-1 <= x1_border) & (x1_border < 0)
        x2_border[m] = 3 * x1_border[m] + 2
        m = (0 <= x1_border) & (x1_border < 1)
        x2_border[m] = 3 * x1_border[m] - 2
        m = x1_border >= 1
        x2_border[m] = -3 * x1_border[m] + 4

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]

    @staticmethod
    def sample_tick(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()
        m = margin * np.sqrt(2)

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)
            neg_mask = (
                ((x[:, 0] < 0) & (x[:, 1] >= -2 - x[:, 0] + m) & (x[:, 1] <= -x[:, 0] - m)) |
                ((x[:, 0] >= 0) & (x[:, 1] >= -2 + x[:, 0] + m) & (x[:, 1] <= x[:, 0] - m))
            )
            pos_mask = (
                (x[:, 1] >= -x[:, 0] + m) & (x[:, 1] >= x[:, 0] + m) |
                (x[:, 1] <= -2 - x[:, 0] - m) | (x[:, 1] <= -2 + x[:, 0] - m)
            )
            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        rand_shift = -2 * torch.randint(low=0, high=2, size=(num_samples, ),
                                        device=y.device, dtype=torch.float32)
        x1_border = -2 + 4 * torch.rand(num_samples, device=y.device)
        x2_border = rand_shift + torch.abs(x1_border)

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]

    @staticmethod
    def sample_cross(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()
        r1, r2 = -2 + np.sqrt(2), 2 - np.sqrt(2)
        m = margin

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)

            pos_mask = (
                ((x[:, 0] <= r1 - m) & (x[:, 1] >= r2 + m)) |
                ((x[:, 0] >= r2 + m) & (x[:, 1] >= r2 + m)) |
                ((x[:, 0] <= r1 - m) & (x[:, 1] <= r1 - m)) |
                ((x[:, 0] >= r2 + m) & (x[:, 1] <= r1 - m))
            )
            neg_mask = (
                ((x[:, 0] >= r1 + m) & (x[:, 0] <= r2 - m)) |
                ((x[:, 1] >= r1 + m) & (x[:, 1] <= r2 - m))
            )
            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        segment_id = torch.randint(low=0, high=8, size=(num_samples, ), device=y.device)
        count = torch.bincount(segment_id)
        x1_border = torch.zeros(size=(num_samples, ), device=y.device)
        x2_border = torch.zeros(size=(num_samples,), device=y.device)

        x1_border[(segment_id == 1) | (segment_id == 6)] = r1
        x1_border[(segment_id == 2) | (segment_id == 5)] = r2
        x2_border[(segment_id == 0) | (segment_id == 3)] = r2
        x2_border[(segment_id == 4) | (segment_id == 7)] = r1

        x1_border[(segment_id == 0) | (segment_id == 7)] = \
            -2 + np.sqrt(2) * torch.rand(((count[0] + count[7]).item(), ), device=y.device)
        x1_border[(segment_id == 3) | (segment_id == 4)] = \
            2 - np.sqrt(2) * torch.rand(((count[3] + count[4]).item(),), device=y.device)
        x2_border[(segment_id == 5) | (segment_id == 6)] = \
            -2 + np.sqrt(2) * torch.rand(((count[5] + count[6]).item(),), device=y.device)
        x2_border[(segment_id == 1) | (segment_id == 2)] = \
            2 - np.sqrt(2) * torch.rand(((count[1] + count[2]).item(),), device=y.device)

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]

    @staticmethod
    def sample_union_jack(y, multiview_mask, margin):
        num_samples = y.shape[0]
        pos_samples = (y == 1).sum().item()
        neg_samples = (y == 0).sum().item()
        m, dm = margin, margin * np.sqrt(2)

        # apply rejection sampling
        pos, neg = torch.tensor([]).reshape(0, 2), torch.tensor([]).reshape(0, 2)
        while len(pos) < pos_samples or len(neg) < neg_samples:
            # apply inverse transform for sampling radius
            x = -2 + 4 * torch.rand((num_samples, 2), device=y.device)
            d1, d2 = x[:, 0], -x[:, 0]

            neg_mask = ((x[:, 1] >= m) & (x[:, 1] <= d1 - dm)) | \
                       ((x[:, 0] <= -m) & (x[:, 1] >= d2 + dm)) | \
                       ((x[:, 1] <= -m) & (x[:, 1] >= d1 + dm)) | \
                       ((x[:, 0] >= m) & (x[:, 1] <= d2 - dm))
            pos_mask = ((x[:, 0] >= m) & (x[:, 1] >= d1 + dm)) | \
                       ((x[:, 1] >= m) & (x[:, 1] <= d2 - dm)) | \
                       ((x[:, 0] <= -m) & (x[:, 1] <= d1 - dm)) | \
                       ((x[:, 1] <= -m) & (x[:, 1] >= d2 + dm))

            pos_left = pos_samples - len(pos)
            neg_left = neg_samples - len(neg)
            pos = torch.cat([pos, x[pos_mask][:pos_left]], dim=0)
            neg = torch.cat([neg, x[neg_mask][:neg_left]], dim=0)

        x = torch.zeros(len(y), 2, dtype=torch.float32, device=y.device)
        x[y == 0] = neg
        x[y == 1] = pos

        line_id = torch.randint(0, 4, size=(num_samples, ), device=y.device)
        x1_border = -2 + 4 * torch.rand(num_samples, device=y.device)
        x2_border = -2 + 4 * torch.rand(num_samples, device=y.device)
        x2_border[line_id == 0] = 0  # horizontal
        x1_border[line_id == 1] = 0  # vertical
        x2_border[line_id == 2] = x1_border[line_id == 2]  # diagonal x2 = x1
        x2_border[line_id == 3] = -x1_border[line_id == 3]  # diagonal x2 = -x1

        x[:, 0] = torch.where(multiview_mask, x[:, 0], x1_border)
        x[:, 1] = torch.where(multiview_mask, x[:, 1], x2_border)
        return x[:, 0], x[:, 1]


def generate_data(
    data_protocol, multiview_probs, data_seed=100, num_features=32,
    train_samples=32, test_samples=5000, device=torch.device('cpu')
):
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    y_train = torch.randint(0, 2, size=(train_samples,), dtype=torch.long, device=device)
    y_test = torch.randint(0, 2, size=(test_samples,), dtype=torch.long, device=device)

    X_train = torch.randn(train_samples, num_features, device=device)
    X_test = torch.randn(test_samples, num_features, device=device)

    if len(data_protocol) == 2 and len(multiview_probs) == 3:
        # special case for 2-features experiments (not described in the paper)
        assert sum(multiview_probs) == 1

        def create_mask(num_samples):
            view_id = np.random.choice([0, 1, 2], p=multiview_probs, size=(num_samples,))
            mask = torch.from_numpy(np.stack([
                (view_id == 0) | (view_id == 1),
                (view_id == 0) | (view_id == 2),
            ], axis=0))
            return mask

        train_mask = create_mask(train_samples)
        test_mask = create_mask(test_samples)

    else:
        # regular case with every prob describing one feature
        assert len(multiview_probs) == len(data_protocol)

        def create_mask(num_samples):
            mask = []
            for prob in multiview_probs:
                indices = np.random.choice(
                    np.arange(num_samples),
                    size=int(prob * num_samples), replace=False
                )
                mask += [np.isin(np.arange(num_samples), indices)]

            return torch.from_numpy(np.stack(mask, axis=0))

        train_mask = create_mask(train_samples)
        test_mask = create_mask(test_samples)

    for k, block in enumerate(data_protocol):
        feature_type = block['feature_type']
        method = 'sample_' + feature_type
        try:
            sampling_func = getattr(Sampler, method)
        except AttributeError:
            raise ValueError('Unknown feature type')

        assert len(block['ids']) == 2
        i, j = block['ids']
        x1, x2 = sampling_func(y_train, train_mask[k], block['margin'])

        noise = block.get('noise', 0.0)
        if noise > 0:
            num_elem = min((y_train == 0).sum().item(), (y_train == 1).sum().item())
            num_elem = int(noise * num_elem)

            neg_elem = torch.arange(train_samples)[y_train == 0].numpy()
            neg_elem = np.random.choice(neg_elem, size=num_elem, replace=False)
            neg_elem = torch.from_numpy(neg_elem)

            pos_elem = torch.arange(train_samples)[y_train == 1].numpy()
            pos_elem = np.random.choice(pos_elem, size=num_elem, replace=False)
            pos_elem = torch.from_numpy(pos_elem)

            x1[neg_elem], x1[pos_elem] = x1[pos_elem], x1[neg_elem]
            x2[neg_elem], x2[pos_elem] = x2[pos_elem], x2[neg_elem]

        X_train[:, i] = x1
        X_train[:, j] = x2

        x1, x2 = sampling_func(y_test, test_mask[k], block['margin'])
        X_test[:, i] = x1
        X_test[:, j] = x2

    return X_train, X_test, y_train, y_test, (train_mask, test_mask)
