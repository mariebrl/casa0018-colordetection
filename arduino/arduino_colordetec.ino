#include <Arduino_APDS9960.h>

#include <TensorFlowLite.h>

// including our model in a seperate file to keep the code simpler
#include "model.h"
// using a variety of functions from the TensorFlowLite library and need to reference them here
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
/*
  Object classifier by color
  --------------------------

  Uses RGB color sensor input to Neural Network to classify colors
  Outputs colors

  Hardware: Arduino Nano 33 BLE Sense board.


#include <Wire.h>
#include <Adafruit_TCS34725.h>
#include <Adafruit_GFX.h>
//#include <Adafruit_SSD1306.h>


#define OLED_RESET 4
//Adafruit_SSD1306 display(OLED_RESET);

Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_600MS, TCS34725_GAIN_1X);
*/

#define RED 22

// global variables used for TensorFlow Lite (Micro)
// Define the variables used by TFLite (pointers)
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create a static memory buffer for TFLM, the size may need to
const int kTensorArenaSize = (2 * 1024);
uint8_t tensor_arena[kTensorArenaSize];

// This constant represents the range of x values our model was trained on
const float kXrange = 255.f;

// This constant determines number of inferences to perform across range of x values defined above. 
const int kInferencesPerCycle = 8000;

// A counter to keep track of how many inferences we have performed.
int inference_count = 0;

// array to map gesture index to a name
const char *COLOURS[] = {"BLACK", "BLUE", "BROWN", "GREEN", "GREY", "ORANGE", "PINK", "PURPLE", "RED", "WHITE", "YELLOW"};
#define NUM_CLASSES (sizeof(COLOURS) / sizeof(COLOURS[0]))

// TF_LITE_REPORT_ERROR
#define TF_LITE_REPORT_ERROR

// compteur loop
int compteur = 3;

/*
 * Initialise the sketch - all this code is run once at start-up
 */
void setup() {
  Serial.println("Object classification using RGB color sensor");
  Serial.println("--------------------------------------------");
  Serial.println("Arduino Nano 33 BLE Sense running TensorFlow Lite Micro");
  Serial.println("");

  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
  }

  // initialize the digital Pin as an output
  pinMode(RED, OUTPUT);

  // initialize the pushbutton as an input:
  const int buttonPin = 2; 

 // check if a color reading is available
  while (! APDS.colorAvailable()) {
    delay(5);
  }
  int r, g, b;
  // read the state of the pushbutton value:
  int buttonState = 0;
  buttonState = digitalRead(buttonPin);
  
  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
  if (buttonState == 0) {
    // read the color
    APDS.readColor(r, g, b);
    // print the values
    Serial.print("Red light = ");
    Serial.println(r);
    Serial.print("Green light = ");
    Serial.println(g);
    Serial.print("Blue light = ");
    Serial.println(b);
    Serial.println();
  
  } else {
    // turn LED off:
    digitalWrite(RED, HIGH);
  }  
  // Initialize LED ON
  digitalWrite(RED, LOW);

  // Set up a tflite mirco error reporter to allow us to log / display data to the terminal. 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load in the model we have defined in the model.cpp file and map it into a usable data structure. 
  model = tflite::GetModel(g_model);

  // Check to see if the model is valid using the version function in the library and 
  // if not valid then break out of setup since the code probably will not work!
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations needed
  static tflite::AllOpsResolver resolver;


  // Build an interpreter to run the model with and create a variable that points to the address of that interpreter.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


    // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors. 
  // We have one tensor so our input and output reference item 0.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // initialise inference counter to 0.
  inference_count = 0;
  
}

void loop() {
  //while(compteur > 0){

  /*
   * The Loop has five main steps:
   *  - work out an input x value
   *  - convert that input value to be the same as the model type
   *  - invoke the interpreter to work out an output given an input
   *  - convert the output back into one of the color classes
   */

  int r, g, b;
  int sum;
  
  /*   
  *  Step 1: work out an input x value 
  */
  // Calculate an x value to feed into the model. 
  // Position is a number between 0 and 1 depending on how far through the
  // inference count cycle we are.
  // We then mulitply that by our range factor to map it between 0 and 2PI.
  // (ie the range of possible x values the model was trained on)
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x_val = position * kXrange;

  // check if both color and proximity data is available to sample
  //while (!APDS.colorAvailable() || !APDS.proximityAvailable()) {
    // read the color and proximity sensor
  //  APDS.readColor(r, g, b);
  //  sum = r + g + b;
  //}
  // read the color
  APDS.readColor(r, g, b);
  // print the values
  Serial.print("Red light = ");
  Serial.println(r);
  Serial.print("Green light = ");
  Serial.println(g);
  Serial.print("Blue light = ");
  Serial.println(b);
  Serial.println();

  /*
  *  Step 2: convert that input value to be the same as the model type (quantized)
  */  
  // check if there's an object close and well illuminated enough
  if (sum > 0) {

    float redRatio = r / 255;
    float greenRatio = g / 255;
    float blueRatio = b / 255;

    // input sensor data to model
    input->data.f[0] = redRatio;
    input->data.f[1] = greenRatio;
    input->data.f[2] = blueRatio;

    /*  
    *  Step 3: invoke the interpreter to work out an output given an input 
    */
    // Run inferencing
    TfLiteStatus invoke_status = interpreter->Invoke();
    
    // Check to see if invoke went ok                       
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n",
                          static_cast<double>(x_val));   
      return;
    }

    /* 
    *  Step 4: convert the output back into a number between -1 and 1
    */
    // Output results
    for (int i = 0; i < 10; i++) {
      Serial.print(COLOURS[i]);
      Serial.print(" ");
      Serial.print(int(output->data.f[i] * 100));
      Serial.print("%\n");
    }
    Serial.println();
  }

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;

  /* 
    *  Step 4: convert the output back into a number between -1 and 1
    */
    // Output results
  for (int i = 0; i < 10; i++) {
      Serial.print(COLOURS[i]);
      Serial.print(" ");
      Serial.print(int(output->data.f[i] * 100));
      Serial.print("%\n");
      Serial.print(" ");
    }
  
  Serial.print(" ");
  compteur = compteur - 1;
  

  // Add ddelay
  delay(5000);
}


