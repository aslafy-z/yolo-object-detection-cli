# yolo-object-detection-cli

## Usage

### Docker

```shell
git clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
docker run -it --rm \
  -v $PWD/samples/data:/samples \
  ghcr.io/aslafy-z/yolo-object-detection-cli \
  --source=/samples/shop.mp4
```

### System

```shell
git clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
# optional: create virtualenv
pip install -r requirements.txt
python main.py --source=samples/data/shop.mp4
```
