import torch


class FourierSlicer:
    def __init__(self, size, blocks=None, pres_low_freq=0, mask_mode=False):
        """
        Generate slices with several blocks of frequencies preserved or masked
        Params:
        size: int -- size of images
        blocks: list of (int, int) -- frequency blocks to preserve/mask
                                      each block is (l, r), where l is included, but r is excluded
        pres_low_freq: int -- how many low frequencies to preserve (0 to preserve None)
        mask_mode: bool -- if True, mask given blocks instead of preserving them
        """
        assert size % 2 == 0
        self.size = size
        self.half_size = size // 2
        self.pres_low_freq = pres_low_freq
        self.mask_mode = mask_mode

        if blocks is None:
            blocks = [(i, i + 1) for i in range(size + 1)]
        self.blocks = blocks

        x_freqs = torch.arange(size).reshape(1, -1, 1).repeat(3, 1, size) - self.half_size
        y_freqs = torch.arange(size).reshape(1, 1, -1).repeat(3, size, 1) - self.half_size
        self.dist = x_freqs.abs() + y_freqs.abs()

    def __call__(self, img):
        """
        Params:
        img: torch.Tensor -- a single image (C, H, W) or a batch of images (B, C, H, W) to slice
        Returns a generator over all possible slices
        """
        freqs = torch.fft.fft2(img)
        freqs_rolled = torch.roll(freqs, shifts=(self.half_size, self.half_size), dims=(-2, -1))
        template = torch.clone(freqs_rolled) if self.mask_mode else torch.zeros_like(freqs_rolled)

        if self.pres_low_freq == 0:
            pass
        elif self.pres_low_freq == 1:
            template[..., self.half_size, self.half_size] = \
                freqs_rolled[..., self.half_size, self.half_size]
        else:
            low_mask = self.dist < self.pres_low_freq
            if len(img.shape) == 4:
                low_mask = low_mask.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
            template[low_mask] = freqs_rolled[low_mask]

        for l, r in self.blocks:
            new_freqs_rolled = torch.clone(template)
            mask = (l <= self.dist) & (self.dist < r)
            if len(img.shape) == 4:
                mask = mask.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)

            if self.mask_mode:
                new_freqs_rolled[mask] = 0
            else:
                new_freqs_rolled[mask] = freqs_rolled[mask]
            new_freqs = torch.roll(new_freqs_rolled, shifts=(self.half_size, self.half_size), dims=(-2, -1))
            yield torch.fft.ifft2(new_freqs).real
