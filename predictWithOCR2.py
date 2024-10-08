# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra                                           # A framework for managing complex configurations
import torch                                           # PyTorch, a deep learning library
import easyocr                                         # Optical Character Recognition (OCR) library
import cv2                                             # A computer vision library
from ultralytics.yolo.engine.predictor import BasePredictor          # Imports the BasePredictor class from the Ultralytics YOLO library
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops         # Imports various utilities from the Ultralytics YOLO library
from ultralytics.yolo.utils.checks import check_imgsz                # Imports the function check_imgsz to validate image sizes
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box       # Imports utilities for plotting and saving bounding boxes
import re                                              # For regular expressions
import os                                              # For operating system dependent functionality

# Initialize the OCR reader globally to avoid re-initializing for each frame
reader = easyocr.Reader(['en'])

def getOCR(im, coors, existing_plates, char_length_min=5, char_length_max=8):
    """
    Perform OCR on the cropped image region and store valid license plates.

    Parameters:
    - im (numpy.ndarray): The original image.
    - coors (list): Coordinates [x, y, w, h] of the bounding box.
    - existing_plates (set): Set of already recognized plates to prevent duplicates.
    - char_length_min (int): Minimum number of characters for a valid plate.
    - char_length_max (int): Maximum number of characters for a valid plate.

    Returns:
    - str: The recognized license plate text or an empty string.
    """
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    cropped_im = im[y:h, x:w]  # Crop the image to the specified region.
    conf_threshold = 0.2  # Confidence threshold for OCR

    gray = cv2.cvtColor(cropped_im, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    results = reader.readtext(gray)  # Perform OCR
    ocr_text = ""

    # Define regex patterns based on the correct answers provided
    patterns = [
        r'^[A-Z]\s?\d{3}\s?[A-Z]{2}$',          # e.g., R 183 JF
        r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$',      # e.g., N 894 JV
        r'^\d{2}\s?[A-Z]{2}\s?\d{2}$',         # e.g., 66 HH 07
        r'^[A-Z]\s?\d{3}\s?[A-Z]{2}$',         # e.g., R 197 GB
        r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$',      # Duplicate pattern for consistency
    ]

    for result in results:
        detected_text = result[1].strip().upper()  # Normalize to uppercase and strip spaces

        # Replace common OCR misinterpretations
        detected_text = detected_text.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')

        # Remove any unwanted characters (keep alphanumerics and common separators)
        detected_text = re.sub(r'[^A-Z0-9\- ]', '', detected_text)

        # Remove spaces and hyphens for length checking
        char_count = len(detected_text.replace(' ', '').replace('-', ''))

        # Check character length
        if not (char_length_min <= char_count <= char_length_max):
            continue  # Skip if not within the desired character length

        # Check if the detected text matches any of the defined patterns
        if any(re.match(pattern, detected_text) for pattern in patterns):
            ocr_text = detected_text
            break  # Stop after finding the first valid plate

    if ocr_text:
        # Add the new plate only if it's not already in the file
        if ocr_text not in existing_plates:
            with open("plates.txt", "a") as f:
                f.write(ocr_text + "\n")
            existing_plates.add(ocr_text)  # Update the set to include the new plate

    return ocr_text


class DetectionPredictor(BasePredictor):
    def __init__(self, cfg, existing_plates):
        super().__init__(cfg)
        self.existing_plates = existing_plates  # Set to track existing plates

    def get_annotator(self, img):
        """Defines a method to get an annotator."""
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """Preprocess the image before prediction."""
        img = torch.from_numpy(img).to(self.model.device)   # Convert the image to a PyTorch tensor and move it to the appropriate device
        img = img.half() if self.model.fp16 else img.float()  # Convert to half precision if supported
        img /= 255.0  # Normalize to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        """Postprocess the predictions."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        """Write prediction results."""
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # Expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # Print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # Detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # Write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # Label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # Integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy, self.existing_plates)  # Call the updated getOCR function
                if ocr:
                    label = ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                            imc,
                            file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                            BGR=True)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    """Main prediction function."""
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # Check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    # Load existing plates to prevent duplicates
    if os.path.exists("plates.txt"):
        with open("plates.txt", "r") as f:
            existing_plates = set(line.strip().upper() for line in f)
    else:
        existing_plates = set()

    predictor = DetectionPredictor(cfg, existing_plates)
    predictor()


if __name__ == "__main__":
    predict()
