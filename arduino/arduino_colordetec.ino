/*
  Object classifier by color
  --------------------------

  Uses RGB color sensor input to Neural Network to classify colors
  Outputs colors

  Hardware: Arduino Nano 33 BLE Sense board.

*/

#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <ArduinoBLE.h>

// including our model in a seperate file to keep the code simpler
#include "model.h"
#include "/Users/mariebourel/Documents/Fac/Master 2022_2023/UCL/cours/CASA0018 - Deep Learning for sensor network /Coursework/arduino_ide/arduino_colordetec/model.cc"


// using a variety of functions from the TensorFlowLite library and need to reference them here
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// LED on the board
#define RED 22
#define GREEN 23
#define BLUE 24

// global variables used for TensorFlow Lite (Micro)
// Define the variables used by TFLite (pointers)
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
//tflite::MicroInterpreter* interpreter = nullptr;

tflite::MicroInterpreter* tflInterpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
char *result;
float max_proba;

// Create a static memory buffer for TFLM, the size may need to
const int kTensorArenaSize = (12 * 1024);
uint8_t tensor_arena[kTensorArenaSize];

// This constant determines number of inferences to perform across range of x values defined above. 
const int kInferencesPerCycle = 4;

// A counter to keep track of how many inferences we have performed.
int inference_count = 0;

// array to map color index to a label
//const char *COLOURS[] = {"BLACK", "BLUE", "BROWN", "GREEN", "GREY", "ORANGE", "PINK", "PURPLE", "RED", "WHITE", "YELLOW"};
char *COLOURS[] = {"BLACK", "BLUE", "BROWN", "GREEN", "GREY", "ORANGE", "PINK", "PURPLE", "RED", "WHITE", "YELLOW"};
#define NUM_CLASSES (sizeof(COLOURS) / sizeof(COLOURS[0]))

// TF_LITE_REPORT_ERROR
#define TF_LITE_REPORT_ERROR

// initialize the color values
int r, g, b, c;

// BLE create service and characteristics
const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

BLEService colordetection_service("280B");
//BLEUnsignedCharCharacteristic colordetectionChar("2101", BLERead | BLENotify);
//BLECharCharacteristic colordetectio_Char(deviceServiceCharacteristicUuid, BLERead | BLENotify | BLEWrite);
BLEStringCharacteristic colordetectio_Char(deviceServiceCharacteristicUuid, BLERead | BLENotify | BLEWrite, 7);
//BLEStringCharacteristic colordetectio_Char("2A19", BLERead | BLENotify | BLEWrite, 7);
/*
 * Initialise the sketch - all this code is run once at start-up
 */
void setup() {
  Serial.begin(9600);
  while (!Serial){};

  Serial.println("Object classification using RGB color sensor");
  Serial.println("--------------------------------------------");
  Serial.println("Arduino Nano 33 BLE Sense running TensorFlow Lite Micro");
  Serial.println("");

  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (!APDS.begin()) {
      Serial.print(F("."));
      delay(500);
    }
  }

  if (!BLE.begin()) {
    Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1);
  }

  // initialize the digital Pin as an output
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(BLUE, OUTPUT);

  // Initialize LED ON
  digitalWrite(RED, LOW);
  digitalWrite(GREEN, LOW);
  digitalWrite(BLUE, LOW);

  // Set up a tflite mirco error reporter to allow us to log / display data to the terminal. 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load in the model we have defined in the model.cpp file and map it into a usable data structure. 
  model = tflite::GetModel(models_model_no_quant_tflite);

  // Check to see if the model is valid using the version function in the library and 
  // if not valid then break out of setup since the code probably will not work!
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Initialize Arduino BLE (service and characteristic)
  BLE.setLocalName("ColorDetect");
  BLE.setAdvertisedService(colordetection_service);
  colordetection_service.addCharacteristic(colordetectio_Char);
  
  // add the service and set a value for the characteristic:
  BLE.addService(colordetection_service);
  //colordetectio_Char.writeValue('ff');
  // start advertising
  BLE.advertise();
  
  Serial.println("Bluetooth device active, waiting for connections...");

  // This pulls in all the operation implementations needed
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with and create a variable that points to the address of that interpreter.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflInterpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Create an interpreter to run the model
  //tflInterpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Obtain pointers to the model's input and output tensors. 
  // We have one tensor so our input and output reference item 0.
  input = tflInterpreter->input(0);
  output = tflInterpreter->output(0);
}

void loop() {
  /*
  * The Loop has five main steps:
  *  - work out an input x value
  *  - convert that input value to be the same as the model type
  *  - invoke the interpreter to work out an output given an input
  *  - convert the output back into one of the color classes
  */

  /*  Step 1: work out an input x value */
  // Calculate input values to feed into the model. 
  while(APDS.colorAvailable()){
    // read the color
    APDS.readColor(r, g, b, c);
    // print the values
    Serial.print("Red light = ");
    Serial.println(r);
    Serial.print("Green light = ");
    Serial.println(g);
    Serial.print("Blue light = ");
    Serial.println(b);
    //Serial.print("Light = ");
    //Serial.println(c);
    Serial.print("\n");

    double sum = r + g + b;

    double redRatio = r / sum;
    double greenRatio = g / sum;
    double blueRatio = b / sum;
    
    /*  Step 2: convert that input value to be the same as the model type */
    // check if there's an object close and well illuminated enough
    input->data.f[0] = redRatio;
    input->data.f[1] = greenRatio;
    input->data.f[2] = blueRatio;

    /*  Step 3: invoke the interpreter to work out an output given an input */
    TfLiteTensor* output = tflInterpreter->output(0);
    // Run inferencing
    TfLiteStatus invoke_status = tflInterpreter->Invoke();

    /*  Step 4: convert the output back into a color label */
    // Output results 
    /*
    for (int i = 0; i < 11; i++) {
      Serial.print("result output ");
      Serial.print("\n");
      Serial.print(COLOURS[i]);
      Serial.print(": ");
      Serial.print(output->data.f[i]*100, 6);
      Serial.print("%\n");
    }
    */

    // Print the max probability
    max_proba = output->data.f[0];
    for (int i = 0; i < 11; i++) {
      max_proba = max(output->data.f[i], max_proba);
    }
    for (int i = 0; i < 11; i++) {
      if (output->data.f[i] == max_proba){
        Serial.print("Colour Label: \n");
        result = COLOURS[i];
        Serial.print(COLOURS[i]);
        Serial.print("\n");
      }
    }
    
    // Display result via BLE
    BLEDevice central = BLE.central();
    Serial.println("- Discovering central device...");
    delay(500);

    if (central) {
      Serial.print("Connected to central: ");
      Serial.println(central.address());

      while (central.connected()) {
        //const unsigned char color = (char*) result ;
        Serial.print("Color Class: ");
        Serial.println(max_proba);
        Serial.println(result);
        //colordetectio_Char.writeValue(*result);
        colordetectio_Char.setValue(result);
        if (colordetectio_Char.subscribed()) {
          // set a new value , that well be pushed to subscribed Bluetooth® Low Energy devices
          //colordetectio_Char.writeValue(*result);
        }
        delay(15000);
      }
    }
  }

  
  delay(15000);
  
  Serial.print("Disconnected from central: ");
  //Serial.println(central.address());

  delay(15000); // wait 10 sec
}


