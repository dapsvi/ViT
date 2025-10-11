import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import einops
import matplotlib.pyplot as plt
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)

batch_size = 512

# Create data loader.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# enable cudnn autotuner if available
if device == "cuda":
    torch.backends.cudnn.benchmark = True

# Define model
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_shape = (8, 8)
        self.hidden_dim = 512
        self.L = 12
        self.image_size = (32, 32)
        self.channels = 3
        
        patch_dim = self.channels * self.patch_shape[0] * self.patch_shape[1]
        num_patches = (self.image_size[0] * self.image_size[1]) // (self.patch_shape[0] * self.patch_shape[1])

        self.conv = nn.Conv2d(3, 3, (16, 16), padding='same')
        self.dropout = nn.Dropout(0.3)
        self.hidden_layer = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.4),
        )
        self.learnable_classification_token = nn.Parameter(torch.zeros((1, self.hidden_dim)))
        self.positional_embeddings = nn.Parameter(torch.zeros((num_patches+1, self.hidden_dim)))
        self.encoders = nn.ModuleList([
            TransformerEncoder(self.hidden_dim) for _ in range(self.L)
        ])
        self.final_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, 10)

    def forward(self, x: torch.Tensor):
        # Step 1. Patch extraction

        # x.shape = (batch_size, 1, 28, 28) for FashionMNIST
        # extract patches -> (batch_size, 14*14, 1*2*2)
        x = self.conv(x) # keep the same shape
        patches: torch.Tensor = einops.rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=self.patch_shape[0],
            p2=self.patch_shape[1]
        )

        # reshape to (batch_size, num_patches, hidden_dim) and concatenate the learnable vector token
        hidden: torch.Tensor = self.hidden_layer(patches)
        hidden = self.dropout(hidden)
        lct = self.learnable_classification_token.expand(x.shape[0], -1, -1) # (batch_size, 1, hidden_dim)
        hidden: torch.Tensor = torch.cat([hidden, lct], dim=1) # (batch_size, num_patches + 1, hidden_dim)

        
        # Step 2. Positional embedding

        pe: torch.Tensor = self.positional_embeddings.expand(x.shape[0], -1, -1)
        final_input: torch.Tensor = hidden + pe # dimension stays the same

        
        # Step 3. Process through transformer encoders
        
        encoder_output = final_input
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output)
        
        
        # Step 4. Classification
        
        encoder_output = self.final_layer_norm(encoder_output)
        # extract the first token
        cls_token = encoder_output[:, 0, :] # (batch_size, hidden_dim)
        output = self.classifier(cls_token)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.num_heads = 16
        self.parent_hidden_dim = hidden_dim
        self.subspace_dim = self.parent_hidden_dim // self.num_heads
        self.QKV = nn.Linear(self.parent_hidden_dim, 3*self.parent_hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm1 = nn.LayerNorm(self.parent_hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.parent_hidden_dim)
        self.MLP = nn.Sequential(
            nn.Linear(self.parent_hidden_dim, self.parent_hidden_dim * 4), # expansion
            nn.GELU(),
            nn.Linear(self.parent_hidden_dim * 4, self.parent_hidden_dim)
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        # x.shape = (batch_size, num_patches + 1, hidden_dim)
        normalized = self.layer_norm1(x)

        QKV: torch.Tensor = self.QKV(normalized) # (batch_size, num_patches + 1, 3 * hidden_dim)
        
        # extract QKV vectors
        Q, K, V = einops.rearrange(
            QKV, 
            'b s (qkv h d) -> qkv b h s d', 
            qkv=3, # split into 3 tensors: Q, K, V
            h=self.num_heads,
            d=self.subspace_dim
        )

        S = x.shape[1] # sequence length : num_patches + 1
        
        # Attention : (batch_size, num_heads, S, S)
        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.subspace_dim))
        attention_weights = self.softmax(scores)
        
        # output head : (batch_size, num_heads, S, subspace_dim)
        output_head = attention_weights @ V
        
        # output : (batch_size, S, hidden_dim)
        output = einops.rearrange(
            output_head,
            "b n S s -> b S (n s)"
        )
        output_projection = self.output_projection(output) # shape remains the same
        x = x + output_projection
        
        # final output
        normalized2 = self.layer_norm2(x)
        MLP_out = self.dropout(self.MLP(normalized2))      # shape remains the same
        x = x + MLP_out
        
        return x



model = ViT().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)




def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, max=None):
    if max is None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
    else:
        size = max
        num_batches = max // batch_size
    model.eval()
    test_loss, correct = 0, 0
    k = 0
    with torch.no_grad():
        total_samples = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            k += 1
            total_samples += len(X)
            if max is not None and total_samples >= max:
                break
    test_loss /= num_batches
    total_samples = k * batch_size
    correct /= total_samples
    print(f"Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss

epochs = 200
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

test_acc = []
train_acc = []
train_loss = []

for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)

    print("Validating test dataset...")
    test_res = test(test_dataloader, model, loss_fn, max=512)
    test_acc.append(test_res[0])
    print("Validating train dataset...")
    train_res = test(train_dataloader, model, loss_fn, max=512)
    train_acc.append(train_res[0])
    train_loss.append(train_res[1])
    
    scheduler.step()



X = np.arange(1, len(test_acc)+1)


test_acc = np.array(test_acc)
train_acc = np.array(train_acc)
train_loss = np.array(train_loss)


fig, ax = plt.subplots(2)

ax[0].plot(X, test_acc, color='green', label='test accuracy')
ax[0].plot(X, train_acc, color='red', label='train accuracy')


ax[1].plot(X, train_loss, color='blue', label='train loss')

ax[0].legend()
ax[1].legend()
fig.tight_layout()
plt.show()