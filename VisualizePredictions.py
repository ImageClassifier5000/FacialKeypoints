import random

import matplotlib.pyplot as plt
import torch

from FacialKeypointsDataset import FacialKeypointsDataset, FacialKeypointsImageDataset
from KeypointsNetwork import KeypointsNetwork


NUM_IMAGES = 5
HIGHLIGHTED_KEYPOINT_INDEX = 9


def plot_random_samples_with_keypoints(dataset, num_images=NUM_IMAGES):
    if num_images <= 0:
        raise ValueError("num_images muss größer als 0 sein.")
    if len(dataset) == 0:
        raise ValueError("Das Dataset enthält keine Bilder.")

    image_count = min(num_images, len(dataset))
    selected_indices = random.sample(range(len(dataset)), image_count)
    figure, axes = plt.subplots(
        1,
        image_count,
        figsize=(4 * image_count, 4),
        squeeze=False,
    )

    for axis, sample_index in zip(axes[0], selected_indices):
        image, keypoints = dataset[sample_index]
        image_tensor = torch.as_tensor(image).detach().cpu()
        if image_tensor.min().item() < 0:
            image_tensor = image_tensor * 0.5 + 0.5

        keypoints_tensor = torch.as_tensor(keypoints).reshape(-1, 2).detach().cpu()
        visible_keypoints = keypoints_tensor[
            torch.isfinite(keypoints_tensor).all(dim=1)
        ]

        axis.imshow(image_tensor.squeeze().numpy(), cmap="gray")
        axis.scatter(
            visible_keypoints[:, 0],
            visible_keypoints[:, 1],
            color="red",
            s=18,
            label="Keypoints",
        )
        axis.set_title(f"Bild {sample_index}")
        axis.axis("off")
        axis.legend(loc="upper right", fontsize="small")

    figure.tight_layout()
    plt.show()
    return figure, axes[0]


def main():
    dataset = FacialKeypointsImageDataset("./RealFace", image_size=96)

    image_count = min(NUM_IMAGES, len(dataset))
    selected_indices = random.sample(range(len(dataset)), image_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointsNetwork().to(device)
    model.load_state_dict(
        torch.load("model_weights.pth", map_location=device, weights_only=True)
    )
    model.eval()

    figure, axes = plt.subplots(
        1,
        image_count,
        figsize=(4 * image_count, 4),
        squeeze=False,
    )

    with torch.no_grad():
        for axis, sample_index in zip(axes[0], selected_indices):
            image = dataset[sample_index]
            prediction = model(image.unsqueeze(0).to(device))[0]
            keypoints = prediction.cpu().reshape(15, 2)

            axis.imshow(image.squeeze(0), cmap="gray")
            axis.scatter(
                keypoints[:, 0],
                keypoints[:, 1],
                color="red",
                s=18,
                label="Keypoints",
            )
            # highlighted = keypoints[HIGHLIGHTED_KEYPOINT_INDEX]
            # axis.scatter(
            #     highlighted[0],
            #     highlighted[1],
            #     color="green",
            #     s=32,
            #     label="Keypoint 10",
            # )
            axis.set_title(f"Validierungsbild {sample_index}")
            axis.axis("off")
            axis.legend(loc="upper right", fontsize="small")

    figure.tight_layout()
    plt.show()


main()