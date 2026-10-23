import torch
from torch.utils.data import DataLoader

from FacialKeypointsDataset import FacialKeypointsDataset
from KeypointsNetwork import KeypointsNetwork
from Loss import Loss


def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    keypoint_distances = [[] for _ in range(15)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            total_loss += loss_fn(predictions, labels).item()

            for keypoint_index in range(15):
                x_index = keypoint_index * 2
                y_index = x_index + 1
                valid = (
                    ~torch.isnan(labels[:, x_index])
                    & ~torch.isnan(labels[:, y_index])
                )
                distances = torch.sqrt(
                    (labels[valid, x_index] - predictions[valid, x_index]) ** 2
                    + (labels[valid, y_index] - predictions[valid, y_index]) ** 2
                )
                keypoint_distances[keypoint_index].extend(
                    distances.cpu().tolist()
                )

    mean_loss = total_loss / len(dataloader)
    print(f"Validation loss: {mean_loss:.6f}")

    for keypoint_index, distances in enumerate(keypoint_distances, start=1):
        if distances:
            mean_distance = sum(distances) / len(distances)
            print(
                f"Keypoint {keypoint_index}: mean distance = "
                f"{mean_distance:.4f} pixels, labeled images = {len(distances)}"
            )
        else:
            print(f"Keypoint {keypoint_index}: no labels available, labeled images = 0")


def main():
    dataset = FacialKeypointsDataset("./training/training.csv")
    (training_dataset, remaining_dataset) = dataset.split((80, 20))
    (_, validation_dataset) = remaining_dataset.split((50, 50))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointsNetwork().to(device)
    model.load_state_dict(
        torch.load("model_weights.pth", map_location=device, weights_only=True)
    )

    evaluate(dataloader, model, Loss(), device)


if __name__ == "__main__":
    main()