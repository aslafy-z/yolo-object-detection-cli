# yolo-object-detection-cli

## Usage

### Docker

```shell
git clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples
docker pull ghcr.io/aslafy-z/yolo-object-detection-cli:latest
docker run -it --rm \
  -v $PWD/samples/data:/samples \
  -p 8000:8000 \
  ghcr.io/aslafy-z/yolo-object-detection-cli:latest \
  --source=/samples/shop.mp4
```

### System

```shell
git clone https://github.com/aslafy-z/yolo-object-detection-cli
cd ./yolo-object-detection-cli
git clone https://github.com/aslafy-z/yolo-object-detection-samples ./samples

python -m virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt

python main.py --source=samples/data/shop.mp4
```
