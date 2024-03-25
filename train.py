from collections import Counter
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import string
import matplotlib.pyplot as plt
import wandb
from flash_attn import flash_attn_func

from train_diagnostics import summarize_activations, summarize_weights, summarize_gradients


K = 27 # space is zero
D = 16
B1 = 0.75**2
characters = set(string.ascii_letters[:26]) | {' '}
words = [word for word in Path('/usr/share/dict/words').read_text().splitlines() if set(word) <= characters and len(word) <= D]
print('words in the dataset:', len(words))
print('distribution of lengths:', Counter(map(len, words)))
words = [word.ljust(D) for word in words] # padding with spaces on the right

def encode(word):
    x = torch.tensor([ord(c) - ord('a') + 1 if c != ' ' else 0 for c in word])
    return F.one_hot(x, num_classes=K).float()

def decode(indices):
    return ''.join(chr(i + ord('a') - 1) if i > 0 else ' ' for i in indices)

characters = ''.join(sorted(characters))
assert torch.allclose(encode(characters), torch.eye(K))


class Block(nn.Module):
    def __init__(self, dim=512, num_blocks=1):
        super().__init__()
        self.dim = dim
        self.head = 128
        self.norm = nn.LayerNorm(dim, bias=False)
        self.input = nn.Linear(dim, self.dim*4, bias=False)
        self.output = nn.Linear(self.dim*2, dim, bias=False)

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=(dim*2)**-0.5 * num_blocks**-0.5)

    def forward(self, x):
        N, D, K = x.shape
        q, k, i, t = self.input(self.norm(x)).chunk(4, dim=-1)
        v = i.sigmoid() * i * t
        h = flash_attn_func(q.view(N, D, -1, self.head), k.view(N, D, -1, self.head), v.view(N, D, -1, self.head))
        y = self.output(torch.cat([v, h.view(N, D, K)], dim=-1))
        return y


class Net(nn.Module):
    def __init__(self, K=K, D=D, dim=512, num_blocks=1):
        super().__init__()
        self.time = nn.Linear(1, dim)
        self.position = nn.Embedding(D, dim)
        self.input = nn.Linear(K, dim)
        self.blocks = nn.ModuleList([Block(dim, num_blocks=num_blocks) for _ in range(num_blocks)])
        self.output_norm = nn.LayerNorm(dim, bias=False)
        self.output = nn.Linear(dim, K)

    def forward(self, x, t):
        N, D, K = x.shape
        x = self.input(2 * x - 1) + self.time(2 * t - 1) + self.position(torch.arange(D, device=x.device))
        for block in self.blocks:
            x = x + block(x)
        x = self.output_norm(x)
        x = self.output(x)
        return x


def output(net, state_NDK, t_ND1):
    device = next(net.parameters()).device
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return net(state_NDK.to(device), t_ND1.to(device)).softmax(dim=-1)


def sample(output_dist):
    "sample from the categorical distributions of every token independently"
    rand = torch.rand(len(output_dist), D, 1, device=output_dist.device)
    cdf = output_dist.cumsum(dim=-1).roll(1, dims=(-1))
    cdf[:, :, 0] = 0
    return (cdf <= rand).sum(dim=-1) - 1


def generate(net, B1=B1, T=10):
    device = next(net.parameters()).device
    state = (torch.ones(K, device=device) / K).repeat(T+1, D, 1)
    ix = torch.arange(0, T)
    t = (ix)/T
    alpha = B1 * (2*(ix + 1) - 1)/T**2
    eps = torch.randn(T, D, K, device=device)

    for i in ix.tolist():
        k = sample(output(net, state[[i]], t[[i]].repeat(1, D, 1)))
        e_k = F.one_hot(k, num_classes=K)

        mu = alpha[i] * (K * e_k - 1)
        std = (alpha[i] * K + 1e-6)**0.5
        y = mu + std * eps[i]

        state[i+1] = y.exp() * state[i]
        state[i+1] = state[i+1] / state[i+1].sum(dim=-1, keepdim=True)

    return state

 
def random_loss(net, ex_NDK, B1=B1):
    N, D, K = ex_NDK.shape
    t_ND1 = ex_NDK.new_empty(N, 1, 1).uniform_().repeat(1, D, 1).clamp(1e-6, 1)
    beta_ND1 = B1 * t_ND1**2

    mu_NDK = beta_ND1 * (K * ex_NDK - 1)
    std_ND1 = (beta_ND1 * K + 1e-6)**0.5
    eps_NDK = torch.randn_like(ex_NDK)

    state_NDK = (mu_NDK + std_ND1 * eps_NDK).softmax(dim=-1)
    e_NDK = output(net, state_NDK, t_ND1)

    loss_NDK = K * B1 * t_ND1 * (ex_NDK - e_NDK).pow(2)
    return loss_NDK.sum(dim=-1).mean()
    

@torch.inference_mode()
def test(net):
    net.eval()
    examples, figures, extra = [], [], {}
    for step in range(5):
        states = generate(net)
        s = decode(states[-1].argmax(dim=-1).cpu())

        with summarize_activations(net, infix=['block'], verbose=False) as log:
            random_loss(net, states[[-1]])

        extra.update(log)
        examples.append(s)

        if step == 0:
            fig, axs = plt.subplots(1, len(states), figsize=(20, 3))
            for i, ax in enumerate(axs):
                ax.matshow(states[i].T.cpu(), aspect='auto')
                ax.set_yticks(range(K), characters)
            fig.suptitle(s)
            fig.tight_layout()
            figures.append(wandb.Image(fig))
            plt.close(fig)
    net.train()
    return examples, {'test': figures, **extra}


def make_config():
    return {
        'steps': 100000,
        'num_blocks': 8,
        'lr': 2e-4,
        'batch_size': 4096,
        'dim': 768,
        'D': 16,
        'K': 27,
        'B1': 0.75**2
    }


def train():
    if not wandb.run:
        wandb.init(project='bfn', config=make_config(), save_code=True)
    conf = wandb.config

    device = 'cuda'
    net = Net(dim=conf.dim, num_blocks=conf.num_blocks).to(device)
    net = torch.compile(net)
    net.train()
    print(net, sum(p.numel() for p in net.parameters()))
    wandb.config.parameters = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / conf.steps)

    dataset = torch.stack([encode(word) for word in words]).to(device)
    
    for step in range(conf.steps):
        optimizer.zero_grad(set_to_none=True)
        batch_indices = torch.randint(0, len(words), (conf.batch_size,))
        #batch_indices = range(conf.batch_size)
        ex_NDK = dataset[batch_indices]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = random_loss(net, ex_NDK).mean()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        if step % 1000 == 0:
            examples, log = test(net)
            log.update(summarize_weights(net))
            log.update(summarize_gradients(net))
        else:
            examples, log = '', {}

        if step % 100 == 0:
            print(f'step={step}', f'loss={loss.item():.3f}', f'grad_norm={grad_norm.item():.3f}', f'lr={lr:.7f}', examples)

        wandb.log({'loss': loss.item(), 'grad_norm': grad_norm.item(), 'lr': lr, **log})
        scheduler.step()

    examples, log = test(net)
    wandb.log(log)
    print(examples)

    torch.save(net.state_dict(), 'model.pt')
    wandb.save('model.pt', policy='end')


if __name__ == '__main__':
    with wandb.init(project='bfn', config=make_config()) as run:
        train()
    if False:
        sweep_configuration = {
            "name": "16chars",
            "method": "grid",
            "metric": {"goal": "minimize", "name": "loss"},
            "parameters": {
                "steps": {"values": [1000, 10000, 100000]},
                "num_blocks": {"values": [4,8]},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="bfn")
        wandb.agent(sweep_id, function=train)
