from evaluate import load_vocabulary, load_model, caption_image
from utils.CheckpointUtils import load_checkpoint
from utils.ImageTransormation import transform


def caption(image_path):
    # Caption
    image = transform(image_path).unsqueeze(0)
    predicted_caption = caption_image(image, model, vocabulary)
    print(predicted_caption)


if __name__ == "__main__":
    checkpoint = load_checkpoint("coco.big.tar", 'cpu')

    vocabulary = load_vocabulary(checkpoint)
    model = load_model(checkpoint, vocabulary)
    model.eval()

    # Good
    caption("data/test2017/000000000016.jpg")
    caption("data/test2017/000000000057.jpg")
    caption("data/val2017/000000226662.jpg")
    caption("data/val2017/000000006763.jpg")

    # Semi Error
    caption("data/test2017/000000000019.jpg")
    caption("data/val2017/000000255965.jpg")
    caption("data/val2017/000000561256.jpg")
    caption("data/val2017/000000558073.jpg")
    caption("data/val2017/000000153217.jpg")
    caption("data/val2017/000000069213.jpg")
    caption("data/val2017/000000443303.jpg")
    caption("data/val2017/000000274219.jpg")

    # Not fully captioned
    caption("data/test2017/000000000069.jpg")
    caption("data/val2017/000000190236.jpg")
    caption("data/val2017/000000412894.jpg")
    caption("data/val2017/000000224724.jpg")
    caption("data/val2017/000000419974.jpg")

    # Completely wrong
    caption("data/val2017/000000046378.jpg")
    caption("data/val2017/000000163562.jpg")
