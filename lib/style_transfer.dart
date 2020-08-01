//import 'dart:io';  // File
import 'dart:math';

import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

abstract class StyleTransfer {
  final String _modelFile = 'magenta_arbitrary-v1-256_int8_prediction_1.tflite';

  // TensorFlow Lite Interpreter object
  Interpreter _interpreter;

  List<int> _inputShape;
  List<int> _outputShape;

  TensorImage _inputImage;
  TensorBuffer _outputBuffer;

  TfLiteType _outputType = TfLiteType.uint8;

//  var _probabilityProcessor;

  NormalizeOp get preProcessNormalizeOp;
  NormalizeOp get postProcessNormalizeOp;

  StyleTransfer() {
    // Load model when the classifier is initialized.
    _loadModel();
  }

  void _loadModel() async {
    try {
      // Creating the interpreter using Interpreter.fromAsset
      _interpreter = await Interpreter.fromAsset(_modelFile);
      print('Interpreter loaded successfully');
      _inputShape = _interpreter.getInputTensor(0).shape;
      _outputShape = _interpreter.getOutputTensor(0).shape;
      _outputType = _interpreter.getOutputTensor(0).type;

      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
//      _probabilityProcessor = TensorProcessorBuilder().add(
//          postProcessNormalizeOp).build();
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  TensorImage _preProcess() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
          _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(preProcessNormalizeOp)
        .build()
        .process(_inputImage);
  }

  void predict(Image image){
    if (_interpreter == null) {
      throw StateError('Cannot run inference, Intrepreter is null');
    }
    _inputImage = TensorImage.fromImage(image);
    _inputImage = _preProcess();
    _interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer());
    print(_outputBuffer.getShape());
    print(_outputBuffer.getDataType());
  }

//  List<double> execute(String contentImagePath, String styleImageName) {
//  List<double> execute(String contentImagePath) {
//
//    var imageFile = new File(contentImagePath);
//    // Initialization code
//    // Create an ImageProcessor with all ops required. For more ops, please
//    // refer to the ImageProcessor Ops section in this README.
////    ImageProcessor imageProcessor = ImageProcessorBuilder()
////        .add(ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOUR))
////        .build();
//
//    // Create a TensorImage object from a File
//    TensorImage tensorImage = TensorImage.fromFile(imageFile);
//
//    // Preprocess the image.
//    // The image for imageFile will be resized to (224, 224)
//    tensorImage = imageProcessor.process(tensorImage);
//
//    // The run method will run inference and
//    // store the resulting values in output.
////    _interpreter.run(input, output);
//
//    return [1.5];
//  }
}

class TFStyleTransfer extends StyleTransfer {
  TFStyleTransfer() : super();

//  @override
//  String get modelName => 'mobilenet_v1_1.0_224_quant.tflite';

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 255);
}
