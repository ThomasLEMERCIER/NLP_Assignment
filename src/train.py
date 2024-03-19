
def train_epoch(model, train_dl, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for data in train_dl:
        sentence_input, subject_input, category, polarity = data.values()
        sentence_input = {k: v.to(device) for k, v in sentence_input.items()}
        subject_input = {k: v.to(device) for k, v in subject_input.items()}
        category = category.to(device)
        target = polarity.to(device)
        
        optimizer.zero_grad()
        outputs = model(sentence_input, subject_input, category)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (outputs.argmax(1) == target).sum().item()
    return running_loss/len(train_dl), running_acc/len(train_dl)


def train(model, train_dl, optimizer, criterion, scheduler, device, n_epochs):
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{n_epochs} loss: {train_loss:.4f} acc: {train_acc:.4f}")
        scheduler.step()
    print("Finished Training")
