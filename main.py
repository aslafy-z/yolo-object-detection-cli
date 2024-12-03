from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from abc import ABC, abstractmethod
import paho.mqtt.client as mqtt
import json
import base64
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from ultralytics.data import load_inference_source
from ultralytics.data.utils import IMG_FORMATS
import logging
import time
import argparse
import asyncio
import signal
import dataclasses
from typing import List, Tuple, Any
from collections import defaultdict

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@dataclasses.dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    tracking_id: int

@dataclasses.dataclass
class Frame:
    original_image: np.ndarray
    annotated_image: np.ndarray
    detections: List[Detection]
    timestamp: float

class DetectionModel:
    def __init__(self, model_path):
        self.track_history = defaultdict(lambda: [])
        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded {model_path} model")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame) -> Tuple[List[Detection], np.ndarray]:
        results = self.model.track(source=frame, persist=True, verbose=False)[0]
        annotated_frame = results.plot(color_mode="instance")
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            tracking_id = int(box.id[0].cpu().numpy() if box.id else -1)

            if tracking_id != -1:
                # update the tracking history
                track = self.track_history[tracking_id]
                track.append((float((x1 + x2) / 2), float((y1 + y2) / 2)))
                if len(track) > 30:
                    track.pop(0)
                # draw the tracking line
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=colors(tracking_id, True), thickness=5)

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                    class_name=self.get_class_name(cls),
                    tracking_id=tracking_id,
                )
            )

        return detections, annotated_frame

    def get_class_name(self, class_id):
        return self.model.names[int(class_id)]

class VideoSource:
    def __init__(self, source: str, image_refresh_rate: int = 5):
        self.source = source
        self.image_refresh_rate = image_refresh_rate
        self.last_refresh = 0
        self.current_frame = None

        if source.lower().endswith(tuple(IMG_FORMATS)):
            self.type = 'image'
        else:
            self.type = 'video'
            print(source)
            self.gen = iter(load_inference_source(source))

    def read(self) -> np.ndarray:
        current_time = time.time()

        if self.type == 'image':
            logging.debug(f"Reading image from {self.source}")
            if current_time - self.last_refresh >= self.image_refresh_rate:
                self.current_frame = cv2.imread(self.source)
                self.last_refresh = current_time
            return True, self.current_frame.copy()
        else:
            logging.debug(f"Reading video frame from {self.source}")
            try:
                _, frames, _ = next(self.gen)
                return True, frames[0]
            except StopIteration:
                return False, None

    def release(self):
        if self.type != 'image':
            self.gen.close()

class OutputHandler(ABC):
    def __init__(self, _: argparse.Namespace):
        pass
    async def initialize(self):
        pass
    @abstractmethod
    async def publish(self, frame_data: str):
        pass
    async def terminate(self):
        pass

class MQTTOutput(OutputHandler):
    def __init__(self, args : argparse.Namespace):
        self.client = mqtt.Client()
        self.topic = args.mqtt_topic
        self.client.connect(args.mqtt_host, args.mqtt_port, 60)
        self.client.loop_start()

    async def publish(self, frame: Frame):
        message = {
            "timestamp": frame.timestamp,
            "detections": frame.detections
        }
        self.client.publish(self.topic, json.dumps(message, cls=EnhancedJSONEncoder))

    async def terminate(self):
        self.client.loop_stop()
        self.client.disconnect()

class ConsoleOutput(OutputHandler):
    async def initialize(self):
        logging.info("Console output initialized.")
    async def publish(self, frame: Frame):
        for d in frame.detections:
            logging.info(
                f"Detected - "
                f"ID: {d.tracking_id}"
                f", Object: {d.class_name}"
                f", Confidence: {d.confidence:.2f}"
                f", BBox: [{d.bbox[0]}, {d.bbox[1]}, {d.bbox[2]}, {d.bbox[3]}]"
            )

class FastAPIWebSocketOutput(OutputHandler):
    def __init__(self, args : argparse.Namespace):
        self.active_connections: List[WebSocket] = []
        app = FastAPI()
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
                    await asyncio.sleep(0)
            except WebSocketDisconnect:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
        @app.get("/", response_class=HTMLResponse)
        async def root_endpoint():
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Object Detection Viewer</title>
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-gray-100 p-6">
                <div class="flex flex-row items-start space-x-8">
                    <div class="flex flex-col items-center space-y-8">
                        <div class="p-4 bg-white rounded-lg shadow-md w-full">
                            <h3 class="text-lg font-semibold">Informations</h3>
                            <p class="text-gray-700">Time: <span id="info-timestamp">0</span></p>
                            <p class="text-gray-700">Messages per second: <span id="info-mps">0</span>/s</p>
                        </div>
                        <div class="image-container text-center bg-white p-4 rounded-lg shadow-md">
                            <h3 class="text-lg font-semibold mb-4">Annotated Frame</h3>
                            <img id="annotated-image" class="max-w-md max-h-96 border-4 border-gray-800 rounded-md mb-8 cursor-pointer" src="" alt="Original Frame" onclick="fullscreen(this)">
                        </div>
                    </div>
                    <div class="details-container bg-white p-6 rounded-lg shadow-md w-full max-w-4xl">
                        <h3 class="text-xl font-semibold mb-4">Detections</h3>
                        <table id="detections-table" class="w-full table-auto border-collapse">
                            <thead>
                                <tr class="bg-gray-200">
                                    <th class="border border-gray-300 px-4 py-2">Tracking ID</th>
                                    <th class="border border-gray-300 px-4 py-2">Class Name</th>
                                    <th class="border border-gray-300 px-4 py-2">Confidence</th>
                                    <th class="border border-gray-300 px-4 py-2">Bounding Box (x1, y1, x2, y2)</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>
                <script>
                    const ws = new WebSocket('$WEBSOCKET_ENDPOINT');
                    let messageCount = 0;
                    let startTime = Date.now();

                    ws.onmessage = (event) => {
                        ws.send('ack');  // Acknowledge message
                        const data = JSON.parse(event.data);

                        // Update info panel
                        const infoTimestampElement = document.getElementById('info-timestamp');
                        infoTimestampElement.textContent = new Date(data.timestamp * 1000).toLocaleString();

                        // Update image
                        const annotatedImageElement = document.getElementById('annotated-image');
                        annotatedImageElement.src = 'data:image/jpeg;base64,' + data.annotated_frame_data_b64;

                        // Update detections
                        const detectionsTable = document.getElementById('detections-table').getElementsByTagName('tbody')[0];
                        detectionsTable.innerHTML = '';  // Clear previous detections

                        data.detections.forEach(detection => {
                            const row = detectionsTable.insertRow();

                            const trackingIDCell = row.insertCell(0);
                            const classNameCell = row.insertCell(1);
                            const confidenceCell = row.insertCell(2);
                            const bboxCell = row.insertCell(3);

                            trackingIDCell.textContent = detection.tracking_id;
                            classNameCell.textContent = detection.class_name;
                            confidenceCell.textContent = (detection.confidence * 100).toFixed(2) + '%';
                            bboxCell.textContent = `(${detection.bbox.join(', ')})`;
                        });

                        messageCount++;
                    };

                    ws.onerror = (error) => {
                        console.error('WebSocket Error:', error);
                    };

                    ws.onclose = () => {
                        console.log('WebSocket connection closed.');
                    };

                    function fullscreen(docElm) {
                        var isInFullScreen = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement;
                        if (!isInFullScreen) {
                            (docElm.requestFullscreen || docElm.mozRequestFullScreen || docElm.webkitRequestFullScreen || docElm.msRequestFullscreen).call(docElm);
                        } else {
                            (document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen).call(document);
                        }
                    }

                    setInterval(() => {
                        const elapsedTime = (Date.now() - startTime) / 1000; // Calculate elapsed time in seconds
                        const messagesPerSecond = messageCount / elapsedTime;
                        const infoMpsElement = document.getElementById('info-mps');
                        infoMpsElement.textContent = messagesPerSecond.toFixed(2);
                        // Reset the counter and start time
                        messageCount = 0;
                        startTime = Date.now();
                    }, 1000);
                </script>
            </body>
            </html>
            """.replace('$WEBSOCKET_ENDPOINT', 'ws://%s:%d%sws' % (args.http_host, args.http_port, args.http_root))
            return HTMLResponse(content=html_content, status_code=200)
        config = uvicorn.Config(app, args.http_host, args.http_port)
        self.server = uvicorn.Server(config)

    async def initialize(self):
        await self.server.serve()

    async def publish(self, frame: Frame):
        annotated_frame_data_b64 = base64.b64encode(cv2.imencode('.jpg', frame.annotated_image)[1].tobytes()).decode('utf-8')
        message = json.dumps({
            "timestamp": frame.timestamp,
            "annotated_frame_data_b64": annotated_frame_data_b64,
            "detections": frame.detections,
        }, cls=EnhancedJSONEncoder)
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                await asyncio.sleep(0)
            except WebSocketDisconnect:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    async def terminate(self):
        self.active_connections.clear()
        # uvicorn handles ctrl+c signal, so we can ignore it
        pass

class DetectionApp:
    def __init__(self, args):
        self.model = DetectionModel(args.model)
        self.source = VideoSource(args.source, args.image_refresh_rate)
        self.args = args
        self.stop_processing: bool = False
        self.output_handlers: List[OutputHandler] = []

        self.output_handlers.append(ConsoleOutput(args))
        if args.http: self.output_handlers.append(FastAPIWebSocketOutput(args))
        if args.mqtt: self.output_handlers.append(MQTTOutput(args))

    def process_frame(self, frame):
        model_detections, annotated_frame = self.model.detect(frame)
        return Frame(
            original_image=frame,
            annotated_image=annotated_frame,
            detections=model_detections,
            timestamp=time.time()
        )

    async def process_frames(self):
        while not self.stop_processing:
            # Read a single frame
            logging.info("Reading frame.")
            ret, original_frame = self.source.read()
            if not ret:
                logging.info("No frame captured.")
                return
            logging.info("Captured frame.")

            frame_data = self.process_frame(original_frame)
            await asyncio.gather(*[handler.publish(frame_data) for handler in self.output_handlers])
            await asyncio.sleep(0)

    async def run(self):
        for handler in self.output_handlers:
            asyncio.create_task(handler.initialize())
        await self.process_frames()

    async def stop(self):
        self.stop_processing = True
        await asyncio.gather(*[handler.terminate() for handler in self.output_handlers])
        logging.info("Terminated output handlers.")
        self.output_handlers.clear()
        self.source.release()

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Application')
    parser.add_argument('--source', type=str, default='0',
                        help='Source for detection: camera index, video file, image file, or URL')
    parser.add_argument('--model', type=str, default="weights/yolo11n.pt",
                        help='Path to model weights (.pt)')
    parser.add_argument('--image-refresh-rate', type=int, default=5,
                        help='Refresh rate in seconds for static images')
    parser.add_argument('--mqtt', action='store_true',
                        help='Enable MQTT publishing')
    parser.add_argument('--mqtt-host', type=str, default='127.0.0.1',
                        help='MQTT broker host')
    parser.add_argument('--mqtt-port', type=int, default=1883,
                        help='MQTT broker port')
    parser.add_argument('--mqtt-topic', type=str, default='detections',
                        help='MQTT broker port')
    parser.add_argument('--http', action='store_true', default=True,
                        help='Enable HTTP publishing')
    parser.add_argument('--http-host', type=str, default='127.0.0.1',
                        help='HTTP server host')
    parser.add_argument('--http-port', type=int, default=8000,
                        help='HTTP server port')
    parser.add_argument('--http-root', type=str, default='/',
                        help='HTTP root')
    return parser.parse_args()

async def run():
    args = parse_args()
    app = DetectionApp(args)

    try:
        await app.run()
    except asyncio.CancelledError:
        await app.stop()
if __name__ == "__main__":
    asyncio.run(run())
