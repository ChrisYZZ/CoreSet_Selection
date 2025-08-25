import time
import torch
import torch.optim as optim
import torch.nn as nn

# Parameters
NUM_NEG = 4
EPOCHS = 20
LR = 0.001

def train_model(model, dataloader, num_items, epochs=EPOCHS, lr=LR, num_neg=NUM_NEG):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, l in dataloader:  # Positive batch
            # Positive loss
            pred_pos = model(u, i)
            loss_pos = criterion(pred_pos, l)

            # Negative sampling (random items; approx unobserved in sparse data)
            num_neg_samples = len(u) * num_neg
            neg_i = torch.randint(0, num_items, (num_neg_samples,))
            neg_u = u.repeat_interleave(num_neg)
            neg_l = torch.zeros(num_neg_samples)
            pred_neg = model(neg_u, neg_i)
            loss_neg = criterion(pred_neg, neg_l)

            loss = loss_pos + loss_neg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}: Loss {total_loss / len(dataloader)}')
    train_time = time.time() - start
    return model, train_time