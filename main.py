from abc import ABC, abstractmethod
import json
import base64
from pathlib import Path
import logging
import time
import argparse
import asyncio
import signal
import dataclasses
from typing import List, Tuple
from collections import defaultdict
import os
import shutil

import cv2
from skimage import io as skimageio
import threading
from threading import Lock as ThreadLock

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from ultralytics.data import load_inference_source
from ultralytics.data.utils import IMG_FORMATS
import torch


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from fastapi.responses import HTMLResponse
import uvicorn

import paho.mqtt.client as mqtt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


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
    def __init__(self, model: str, model_dir: str, export_dir: str, device: str):
        logging.info(f"Cuda is available: {torch.cuda.is_available()}")
        self.track_history = defaultdict(lambda: [])

        try:
            is_gpu = device != "cpu"
            model_path = Path(model_dir) / model
            exported_model_path = Path(export_dir) / model_path.with_suffix('.engine').name

            if is_gpu and not Path(exported_model_path).exists():
                logging.info(f"Exporting model for GPU usage: {exported_model_path}")
                temp_export_path = YOLO(model_path).export(
                    format="engine", device=device, half=True
                )
                os.makedirs(os.path.dirname(exported_model_path), exist_ok=True)
                shutil.move(temp_export_path, exported_model_path)

            self.model = YOLO(exported_model_path if is_gpu else model_path)
            logging.info(f"Successfully loaded model from {exported_model_path if is_gpu else model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Error initializing YOLO model: {e}")

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
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=colors(tracking_id, True),
                    thickness=5,
                )

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


class ImageSource:
    def __init__(self, source):
        self.source = source
        self.fps = None

    def read(self):
        logging.debug(f"Reading image frame from {self.source}")
        try:
            frame = skimageio.imread(self.source)
            return True, frame.copy()
        except:
            return False, None

    def close():
        pass


class VideoSource:
    def __init__(self, source):
        self.source = source
        self.loader = load_inference_source(source)
        self.gen = iter(self.loader)
        self.fps = None

    def read(self):
        if hasattr(self.loader, "fps") and self.loader.fps:
            self.fps = (
                self.loader.fps[0]
                if isinstance(self.loader.fps, list)
                else self.loader.fps
            )
        logging.debug(f"Reading video frame from {self.source}")
        try:
            _, frames, _ = next(self.gen)
            return True, frames[0].copy()
        except StopIteration:
            return False, None

    def close(self):
        if self.gen and hasattr(self.gen, "close"):
            self.gen.close()


class FrameSource:
    def __init__(self, source: str, frame_interval: int):
        if source.lower().endswith(tuple(["." + a for a in IMG_FORMATS])):
            self.source = ImageSource(source)
        else:
            self.source = VideoSource(source)

        self.frame_interval_seconds = frame_interval / 1000.0

        self.last_capture_time = 0
        self.last_return_time = 0
        self.current_frame = None
        self.last_returned_frame = None

        self.thread_lock = ThreadLock()
        self.thread_should_stop = threading.Event()
        self.thread = threading.Thread(
            target=self._start_background_capture, daemon=True
        )
        self.thread.start()

    def _start_background_capture(self):
        """
        Continuously reads frames from the source at the original FPS.
        """
        while not self.thread_should_stop.is_set():
            current_time = time.time()
            # determine frame read interval based on video FPS
            read_interval = (
                1 / self.source.fps if self.source.fps else 0.033
            )  # Default to ~30 FPS
            # read a new frame if the interval has elapsed
            if current_time - self.last_capture_time >= read_interval:
                ret, frame = self.source.read()
                if ret:
                    with self.thread_lock:
                        self.current_frame = frame
                    self.last_capture_time = current_time
            time.sleep(0.001)

    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Returns the current frame only if the user-defined refresh rate interval has elapsed.
        If called before the interval is elapsed, it returns the previously returned frame.
        """
        current_time = time.time()
        with self.thread_lock:
            if current_time - self.last_return_time >= self.frame_interval_seconds:
                # Enough time has passed; return the latest frame
                self.last_returned_frame = self.current_frame
                self.last_return_time = current_time
                return True, self.last_returned_frame
            else:
                # Return the previous frame if interval hasn't elapsed
                return True, (
                    self.last_returned_frame
                    if self.last_returned_frame is not None
                    else self.current_frame
                )

    def release(self):
        self.thread_should_stop.set()
        self.thread.join()
        self.source.close()


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
    def __init__(self, args: argparse.Namespace):
        self.client = mqtt.Client()
        self.topic = args.mqtt_topic
        self.client.connect(args.mqtt_host, args.mqtt_port, 60)
        self.client.loop_start()

    async def publish(self, frame: Frame):
        message = {"timestamp": frame.timestamp, "detections": frame.detections}
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
    def __init__(self, args: argparse.Namespace):
        self.active_connections: List[WebSocket] = []
        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(connection: WebSocket):
            await connection.accept()
            self.active_connections.append(connection)
            try:
                while True:
                    await connection.receive_text()
            except (
                WebSocketDisconnect,
                uvicorn.protocols.utils.ClientDisconnected,
                ConnectionClosedError,
                ConnectionClosedOK,
            ):
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

        @app.get("/", response_class=HTMLResponse)
        async def root_endpoint():
            html_content = r"""
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
                    const ws = new WebSocket(((window.location.protocol === "https:") ? "wss://" : "ws://") + window.location.host + window.location.pathname.replace(/\/$/, "") + "/ws");
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
            """
            return HTMLResponse(content=html_content, status_code=200)

        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=args.http_host,
                port=args.http_port,
                # Important: use loop to ensure it works with asyncio
                loop="asyncio",
                # Disable lifespan to prevent blocking
                lifespan="off",
            )
        )

    async def initialize(self):
        asyncio.create_task(self._run_server())
        await asyncio.sleep(0.1)

    async def _run_server(self):
        await self.server.serve()

    async def publish(self, frame: Frame):
        annotated_frame_data_b64 = base64.b64encode(
            cv2.imencode(".jpg", frame.annotated_image)[1].tobytes()
        ).decode("utf-8")
        message = json.dumps(
            {
                "timestamp": frame.timestamp,
                "annotated_frame_data_b64": annotated_frame_data_b64,
                "detections": frame.detections,
            },
            cls=EnhancedJSONEncoder,
        )
        disconnected_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except (
                WebSocketDisconnect,
                uvicorn.protocols.utils.ClientDisconnected,
                ConnectionClosedError,
                ConnectionClosedOK,
            ):
                disconnected_connections.append(connection)
        for connection in disconnected_connections:
            self.active_connections.remove(connection)

    async def terminate(self):
        for connection in self.active_connections:
            await connection.close()
        self.active_connections.clear()
        # uvicorn handles ctrl+c signal, so we can ignore it
        pass


class DetectionApp:
    def __init__(self, args):
        self.model = DetectionModel(args.model, args.model_dir, args.export_dir, args.device)
        self.source = FrameSource(args.source, args.frame_interval)
        self.args = args
        self.stop_processing: bool = False
        self.output_handlers: List[OutputHandler] = []

        self.output_handlers.append(ConsoleOutput(args))
        if args.http:
            self.output_handlers.append(FastAPIWebSocketOutput(args))
        if args.mqtt:
            self.output_handlers.append(MQTTOutput(args))

    def process_frame(self, frame):
        model_detections, annotated_frame = self.model.detect(frame)
        return Frame(
            original_image=frame,
            annotated_image=annotated_frame,
            detections=model_detections,
            timestamp=time.time(),
        )

    async def process_frames(self):
        previous_frame = None
        while not self.stop_processing:
            await asyncio.sleep(0)
            ret, original_frame = self.source.read()
            if not ret:
                logging.info("No frame captured.")
                return
            if np.array_equal(previous_frame, original_frame):
                continue
            previous_frame = original_frame
            logging.info("Captured frame.")

            frame_data = self.process_frame(original_frame)
            await asyncio.gather(
                *[handler.publish(frame_data) for handler in self.output_handlers]
            )

    async def run(self):
        await asyncio.gather(
            *[handler.initialize() for handler in self.output_handlers]
        )
        await self.process_frames()

    async def stop(self):
        self.stop_processing = True
        await asyncio.gather(*[handler.terminate() for handler in self.output_handlers])
        logging.info("Terminated output handlers.")
        self.output_handlers.clear()
        self.source.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Application")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source for detection (default: '0'). Use '0' for the default webcam, an index (e.g., '1') for additional webcams, or specify a path to a video/image file or URL.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./weights",
        help="Path to the directory containing the model weights file (default: './weights').",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./weights-optimized",
        help="Path to export the optimized model engine file (default: './weights-optimized'). Used for GPU acceleration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to the model weights file (default: 'yolo11n.pt'). Model will be downloaded if not found.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=0,
        help="Minimum interval between consecutive frame outputs in milliseconds (default: 0 for no limit).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (default: 'cpu'). Options: 'cpu', 'cuda', 'cuda:0' (specific GPU), 'dla', or 'dla:0' (specific DLA).",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Enable MQTT publishing (default: disabled).",
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="127.0.0.1",
        help="MQTT broker host (default: '127.0.0.1').",
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883, help="MQTT broker port (default: 1883)."
    )
    parser.add_argument(
        "--mqtt-topic",
        type=str,
        default="detections",
        help="MQTT topic to publish detections (default: 'detections').",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        default=True,
        help="Enable HTTP publishing (default: enabled).",
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default="127.0.0.1",
        help="HTTP server host (default: '127.0.0.1').",
    )
    parser.add_argument(
        "--http-port", type=int, default=8000, help="HTTP server port (default: 8000)."
    )
    parser.add_argument(
        "--http-root", type=str, default="/", help="HTTP root path (default: '/')."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: 'INFO'). Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    args = parser.parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        logging.warning(
            "CUDA is not available. Falling back to CPU for computation."
        )
        args.device = "cpu"
    return args

async def run():
    args = parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.log_level.upper()))

    app = DetectionApp(args)

    try:
        await app.run()
    except asyncio.CancelledError:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(run())
