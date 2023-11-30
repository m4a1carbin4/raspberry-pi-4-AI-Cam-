import torch
import requests
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
import time
from diffusers import LatentConsistencyModelImg2ImgPipeline
from PIL import Image
import traceback, sys
import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSlider,
    QTabWidget,
    QSpacerItem,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import (
    QSize,
    pyqtSignal,
    pyqtSlot,
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    Qt,
)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(camera_config)

GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.IN)

pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
            )

pipe.to(torch_device = 'cpu',torch_dtype=torch.float32)
pipe.enable_attention_slicing()

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("another world cam")
        self.setFixedSize(QSize(800,400))
        self.init_ui()
        self.threadpool = QThreadPool()

    def init_ui(self):
        self.create_settings_tab()
        self.show()
        
    def create_settings_tab(self):

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("add postive prompt : masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo,")
        self.generate = QPushButton("Cam Start")
        self.generate.clicked.connect(self.cam_to_image)
        self.prompt.setFixedHeight(35)

        self.label = QLabel("img")
        self.label.setFixedSize(QSize(480, 320))

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.prompt)
        hlayout.addWidget(self.generate)
        
        self.strength_value = QLabel("Prompt strength : 0.5")
        self.strength = QSlider(orientation=Qt.Orientation.Horizontal)
        self.strength.setMaximum(100)
        self.strength.setMinimum(0)
        self.strength.setValue(50)
        self.strength.valueChanged.connect(self.update_strength_label)
        
        self.inference_steps_value = QLabel("Number of inference steps: 8")
        self.inference_steps = QSlider(orientation=Qt.Orientation.Horizontal)
        self.inference_steps.setMaximum(25)
        self.inference_steps.setMinimum(1)
        self.inference_steps.setValue(8)
        self.inference_steps.valueChanged.connect(self.update_label)
        
        self.guidance_value = QLabel("Guidance scale: 2")
        self.guidance = QSlider(orientation=Qt.Orientation.Horizontal)
        self.guidance.setMaximum(200)
        self.guidance.setMinimum(10)
        self.guidance.setValue(20)
        self.guidance.valueChanged.connect(self.update_guidance_label)

        vlayout = QVBoxLayout()
        vspacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.strength_value)
        vlayout.addWidget(self.strength)
        vlayout.addWidget(self.inference_steps_value)
        vlayout.addWidget(self.inference_steps)
        vlayout.addWidget(self.guidance_value)
        vlayout.addWidget(self.guidance)
        vlayout.addItem(vspacer)

        mainlayout = QHBoxLayout()
        mainlayout.addWidget(self.label)
        mainlayout.addLayout(vlayout)

        self.settings = QWidget()

        self.settings.setLayout(mainlayout)
        
        self.setCentralWidget(self.settings)

    def update_label(self, value):
        self.inference_steps_value.setText(f"Number of inference steps: {value}")

    def update_strength_label(self, value):
        val = round(int(value) / 100, 2)
        self.strength_value.setText(f"Prompt strength : {val}")
        
    def update_guidance_label(self, value):
        val = round(int(value) / 10, 1)
        self.guidance_value.setText(f"Guidance scale: {val}")
        
    def generate_image(self):
        prompt = self.prompt.toPlainText()
        prompt_strength = round(int(self.strength.value()) / 100, 2)
        guidance_scale = round(int(self.guidance.value()) / 10, 1)
        num_inference_steps = self.inference_steps.value()
        
        print("cam start")
        
        picam2.start()

        while True:

            im = picam2.capture_array()
            #im=cv2.imread("image.png")

            img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            h,w,c = img.shape

            qImg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qImg)
            pixmap = pixmap.scaled(480,320,Qt.KeepAspectRatio)

            self.label.setPixmap(pixmap)

            if (GPIO.input(18) == 0):
                print("print start")
                picam2.capture_file("test_input.jpg")
                time.sleep(2)

                image = Image.open("./test_input.jpg")

                result = pipe(
                            prompt=prompt,
                            image=image,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            strength=prompt_strength,
                            output_type="pil",
                        ).images

                result[0].save("image.png")

                res = requests.get("http://localhost:3000")

                print("end job")
            
                break
        
    def cam_to_image(self):
        worker = Worker(self.generate_image)
        self.threadpool.start(worker)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
