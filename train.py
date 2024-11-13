import torch
import numpy as np
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def patch_loss(model, loss_func, X, y: torch.Tensor, optimizer=None):
    predicted = model(X)
    most_popular_class = torch.mode(y.view(y.size(0), -1), dim=1)
    loss_ = loss_func(
       predicted, 
       most_popular_class.values.type(torch.ByteTensor).to(device)
       )
    if optimizer is not None:
      loss_.backward()
      optimizer.step()
      optimizer.zero_grad()

    return loss_.item(), len(X)

def patch_validate(model, loss_func, X, y):
    predicted = model(X)
    most_popular_class = torch.mode(y.view(y.size(0), -1), dim=1)
    ground_truth = most_popular_class.values.type(torch.ByteTensor).to(device)
    loss_ = loss_func(
       predicted, 
       ground_truth
       )
    pred = torch.argmax(predicted, dim=1)
    correct = pred == ground_truth
    return loss_.item(), torch.sum(correct).item(), len(X)

def pixel_validate(model, loss_func, X, y: torch.Tensor):
    predicted = model(X)
    most_popular_class = torch.mode(y.view(y.size(0), -1), dim=1)
    ground_truth = most_popular_class.values.type(torch.ByteTensor).to(device)
    loss_ = loss_func(
       predicted, 
       ground_truth
       )
    pred = torch.argmax(predicted, dim=1)
    accuracy = 0
    for i in range(0, len(y)):
      predicted_class = pred[i]
      accuracy += torch.divide(
         torch.sum(torch.where(y[i] == predicted_class, 1, 0)).type(torch.FloatTensor).to(device), 
         y.size(2) * y.size(3)
         )

    return loss_.item(), accuracy, len(X)

def evaluate(model: torch.nn.Module, loss_func, loader):
    model.eval()

    with torch.no_grad():
        validated_batches = []
        
        i = 0
        for batch in loader:
          i += 1
          for X, y in batch:
            validated_batches.append(patch_validate(model, loss_func, X, y))
          print("evaluate ", i)

        losses, corrects, nums = zip(*validated_batches)
        test_loss = sum(np.multiply(losses, nums)) / sum(nums)
        test_accuracy = sum(corrects) / sum(nums) * 100

    print(f"Test loss: {test_loss:.5f}\t"
          f"Test accruacy: {test_accuracy:.3f}%")
    return test_loss, test_accuracy

def fit(epochs, model, loss_func, optimizer, train_loader, valid_loader, patience=10):
    graphic_losses = []

    wait = 0
    valid_loss_min = np.Inf

    for epoch in tqdm.tqdm(range(epochs)):

        model.train()

        losses = []
      
        for batch in tqdm.tqdm(train_loader):
          for X, y in batch:
            losses.append(patch_loss(model, loss_func, X, y, optimizer))

        losses, nums = zip(*losses)
        train_loss = sum(np.multiply(losses, nums)) / sum(nums)

        model.eval()

        with torch.no_grad():

            losses = []
            for batch in tqdm.tqdm(valid_loader):
              for X, y in batch:
                losses.append(pixel_validate(model, loss_func, X, y))

            torch.save(model.state_dict(), 'model.pt')

            losses, corrects, nums = zip(*losses)
            valid_loss = sum(np.multiply(losses, nums)) / sum(nums)
            valid_accuracy = sum(corrects) / sum(nums) * 100

            print(f"\nepoch: {epoch+1:3}, loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, valid accruacy: {valid_accuracy:.3f}%")

            graphic_losses.append((train_loss, valid_loss, valid_accuracy))

            # Save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                torch.save(model.state_dict(), 'final_model.pt')
                valid_loss_min = valid_loss
                wait = 0
            # Early stopping
            else:
                wait += 1
                if wait >= patience:
                    print(f"Terminated Training for Early Stopping at Epoch {epoch+1}")
                    return graphic_losses

    return graphic_losses
