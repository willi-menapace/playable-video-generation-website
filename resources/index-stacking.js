var frame = null;
var action = null;
var session = null;

var hidden0 = null;
var hidden1 = null;
var hidden2 = null;
var hidden_cell0 = null;
var hidden_cell1 = null;
var hidden_cell2 = null;
 
var initial_frame_index = 0;

async function init() {
  await load_arrays();
  session = new onnx.InferenceSession() //{backendHint: 'cpu'})
  await session.loadModel(model_array);
   
  await reset();
  run(0);
  draw();
  var load = document.getElementById('load');
  load.style.display = "none";
}

function disable_buttons() {
   var inputs = document.getElementsByTagName("button");
   for (var i = 0; i < inputs.length; i++) {
        inputs[i].disabled = true;
   }
}

function enable_buttons() {
   var inputs = document.getElementsByTagName("button");
   for (var i = 0; i < inputs.length; i++) {
        inputs[i].disabled = false;
   }
}

function getRndInteger(min, max) {
  return Math.floor(Math.random() * (max - min + 1) ) + min;
}

function reset_random() {
   initial_frame_index = getRndInteger(0, images_array.length-1);
   reset();
}

async function reset() {
   const preprocessedData = preprocess(images_array[initial_frame_index], W, H, stacking);	
   frame = new onnx.Tensor(preprocessedData, 'float32', [3 * stacking, H, W]);
   
   hidden0 = initial_hidden0;
   hidden1 = initial_hidden1;
   hidden2 = initial_hidden2;

   hidden_cell0 = initial_hidden_cell0;
   hidden_cell1 = initial_hidden_cell1;
   hidden_cell2 = initial_hidden_cell2;
 
   draw(); 
}

async function draw() { 
  var canvas = document.getElementById('screen'); 
  var ctx = canvas.getContext('2d');
  display(ctx);
}

function preprocess(data, width, height, stacking) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 3]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3 * stacking), [3 * stacking, width, height]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // HWC to CHW and stacks the images
  var i;
  for (i = 0; i < stacking * 3; i++) {
    ndarray.ops.assign(dataProcessed.pick(i, null, null), dataFromImage.pick(null, null, i % 3));
  }
  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  //ndarray.ops.assign(dataProcessed.pick(0, null, null), dataFromImage.pick(null, null, 0));
  //ndarray.ops.assign(dataProcessed.pick(1, null, null), dataFromImage.pick(null, null, 1));
  //ndarray.ops.assign(dataProcessed.pick(2, null, null), dataFromImage.pick(null, null, 2));

  return dataProcessed.data;
}

function display(ctx) {
   var imageData = new Float32Array(H * W * 4);
   const image_array = ndarray(imageData, [W, H, 4]);
   const frame_array = ndarray(frame.data, [3 * stacking, W, H]);
   const alpha = ndarray(new Float32Array(H * W).fill(1), [W, H]);


   ndarray.ops.assign(image_array.pick(null, null, 0), frame_array.pick(0, null, null));
   ndarray.ops.assign(image_array.pick(null, null, 1), frame_array.pick(1, null, null));
   ndarray.ops.assign(image_array.pick(null, null, 2), frame_array.pick(2, null, null));
   ndarray.ops.assign(image_array.pick(null, null, 3), alpha);
   
   ndarray.ops.addseq(image_array, 1);
   ndarray.ops.divseq(image_array, 2);
   ndarray.ops.mulseq(image_array, 255);
   
   imageData = new Uint8ClampedArray(imageData);

   myImageData = new ImageData(imageData, W, H)

   var renderer = document.createElement('canvas');
   renderer.width = myImageData.width;
   renderer.height = myImageData.height;
   renderer.getContext('2d').putImageData(myImageData, 0, 0);
   ctx.drawImage(renderer, 0,0, W*2, H*2);
}

async function run(action) {
  disable_buttons();
  array_act = new Array(n_act).fill(0);
  array_act[action] = 1;
  action = new onnx.Tensor(array_act, 'float32', [1, n_act])
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([frame, action, hidden0, hidden1, hidden2, hidden_cell0, hidden_cell1, hidden_cell2]);
  var values = outputMap.values();

  next_frame = values.next().value;
  const old_frame_array = ndarray(frame.data, [3 * stacking, H, W]);
  const next_frame_array = ndarray(next_frame.data, [3 * stacking, H, W]);
  next_frame_stacked_array = ndarray(new Float32Array(W * H * 3 * stacking), [3 * stacking, H, W]);

  var i;
  for (i = 0; i < stacking * 3; i++) {
    if (i < 3) {
      ndarray.ops.assign(next_frame_stacked_array.pick(i, null, null), next_frame_array.pick(i, null, null));
    } else {
      ndarray.ops.assign(next_frame_stacked_array.pick(i, null, null), old_frame_array.pick(i - 3, null, null));
    }
    
  }

  frame = new onnx.Tensor(next_frame_stacked_array.data, 'float32', [3 * stacking, H, W]);

  //frame = values.next().value;
  
  hidden0 = values.next().value;
  hidden1 = values.next().value;
  hidden2 = values.next().value;
  hidden_cell0 = values.next().value;
  hidden_cell1 = values.next().value;
  hidden_cell2 = values.next().value;
  draw();
  enable_buttons();
}
