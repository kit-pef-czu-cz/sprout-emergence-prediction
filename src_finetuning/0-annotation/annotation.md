# Annotation Requirements for Our Segmentation Model
To train and fine-tune the instance segmentation model using Mask R-CNN within Detectron2, specific annotation requirements must be met. This section outlines the necessary steps and tools for creating suitable annotations.

## Required Annotations Format
Type : Polygon Annotations
Format : COCO (Common Objects in Context) JSON

### Annotation Tool
We recommend using Label Studio , an open-source data labeling tool, to create polygon annotations. Label Studio provides flexibility and ease of use, allowing annotators to draw precise polygons around objects of interest.
Another alternative may be labelme. However, you can choose any annotation software that suits you.

### Example of the annotations
Example of our used annotations and their format could be found here: `./emergence-detection/models/segmentation_model/metadata`
If you want to add your own annotation, create a new folder inside segmentation model.


# Annotation Requirements for Our Object Detection Model
To train and fine-tune the object segmentation model using YOLOv8, specific annotation requirements must be met. This section outlines the necessary steps and tools for creating suitable annotations.

## Required Annotations Format
Type : Bounding Box Annotations
Format : YOLO (You Only Look Once) format

### Annotation Tool
We utilized Roboflow , a cloud-based platform for data annotation, to create bounding box annotations. You can choose any annotation software that support YOLO format.

### Example of the annotations
Organize your images and YOLO-formatted annotation files into a structure recognized by YOLOv8's training scripts.
Typically, this involves creating separate directories for images and annotations, often in a format like images/ and labels/.
Example of our used annotations and their format could be found here: `./emergence-detection/models/object_detection_model/detect_annotations`
Utilize YOLOv8’s training scripts by specifying paths to your dataset directories. Configure additional parameters such as class names and model configuration files according to your project's needs.


# Annotation Requirements for Our Time-Series Model
To train and fine-tune the time-series prediction model using TCN on Tensorflow, specific annotation requirements must be met. This section outlines the necessary steps and tools for creating suitable annotations.

## Required Annotations Format
Type : String with information of the time of first occurence
Format : XLSX or CSV

### Example of the annotations
The annotation for creating a NumPy dataset for a time series must be stored in XLSX format, or CSV format if the code is adjusted accordingly. The required columns are:
 - tray_location: The code designation of the seed tray.
 - well_id: The number of the stored box, e.g., 1-5. In this case, 1 represents the row number in the seed tray, and 5 represents the column number.
 - time_of_first_occurrence: The first frame in which the germinated plant appears

We understand that it can be difficult to know exactly what format annotations must be written in. For easier understanding an example of our used annotations and their format could be found here: `./emergence-detection/models/prediction_model/annotation/`
If you want to add your own annotation, simply add a new file inside the folder.
