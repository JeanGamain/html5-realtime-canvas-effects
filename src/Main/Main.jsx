import React from 'react';

import styles from './Main.css';

class Main extends React.Component {
  constructor(props) {
    super(props);

    this.initializeCompution = this.initializeCompution.bind(this);
    this.createBlurMatrix = this.createBlurMatrix.bind(this);
    this.createGPUKernel = this.createGPUKernel.bind(this);
    this.timerCallback = this.timerCallback.bind(this);
    this.computeFrame = this.computeFrame.bind(this);
    this.convolution = this.convolution.bind(this);

    this.video = null;
    this.canvas1 = null;
    this.canvasContext1 = null;
    this.canvas2 = null;
    this.canvasContext2 = null;
    this.blurMatrix = null;
    this.blurGPUKernel = null;
    this.mainImage = null;
    this.doubleBuffer = null;
    this.gpu = new GPU();

    this.state = {
      width: 0,
      height: 0
    };
  }

  componentDidMount() {
    let self = this;

    navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: 720, height: 480 },
      })
      .then(function(stream) {
        // Older browsers may not have srcObject
        if ("srcObject" in video) {
          self.video.srcObject = stream;
        } else {
          // Avoid using this in new browsers, as it is going away.
          self.video.src = window.URL.createObjectURL(stream);
        }
        video.onloadedmetadata = function(e) {
          video.play();
        };
      })
      .catch(function(err) {
        console.log(err.name + ": " + err.message);
      });
  
    self.video.addEventListener("play", function () {
        self.canvasContext1 = self.canvas1.getContext("2d");
        self.canvasContext2 = self.canvas2.getContext("2d");
        self.initializeCompution();
        self.setState({
          width: self.video.videoWidth,
          height: self.video.videoHeight
        });
        self.timerCallback();
      }, false);
  }

  timerCallback() {
    if (this.video.paused || this.video.ended) {
      return;
    }
    this.computeFrame();
    setTimeout(this.timerCallback, 0);
  }

  initializeCompution() {
    this.canvas1.width = this.video.videoWidth;
    this.canvas1.height = this.video.videoHeight;
    this.canvas2.width = this.video.videoWidth;
    this.canvas2.height = this.video.videoHeight;
    this.mainImage = new Float32Array(this.video.videoWidth * this.video.videoHeight);
    this.doubleBuffer = new Float32Array(this.video.videoWidth * this.video.videoHeight);
    this.blurMatrix = this.createBlurMatrix(3);
    this.blurGPUKernel = this.createGPUKernel(this.video.videoWidth, this.video.videoHeight, this.blurMatrix.mean);
  }

  createBlurMatrix(sigma) {
    let size = (2 * Math.floor(2 * sigma) + 3);
    let mean = Math.floor(size / 2);
    let blurMatrix = {
      size: size,
      mean: mean,
      matrix: new Float32Array(size * size)
    };
    let c = 0;
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        blurMatrix.matrix[c++] = Math.exp(-0.5 * (Math.pow((x - mean) / sigma, 2) + Math.pow((y - mean) / sigma, 2))) / (2 * Math.PI * sigma * sigma);
      }
    }
    return (blurMatrix);
  }

  createGPUKernel(width, height, mean) {
    let kernel =
      this.gpu.createKernel(function(a, b) {
        if (this.thread.x < this.constants.minxy || this.thread.x >= this.constants.maxx ||
            this.thread.y < this.constants.minxy || this.thread.y >= this.constants.maxy) {
          return (a[this.thread.y][this.thread.x]);
        }
        var p = 0;
        for (var ky = -this.constants.khalf; ky <= this.constants.khalf; ky++) {
          for (var kx = -this.constants.khalf; kx <= this.constants.khalf; kx++) {
            p += a[this.thread.y - ky][this.thread.x - kx] * b[ky][kx];
          }
        }
        return p;
      },
      {
        constants: {
          x: width,
          y: height,
          minxy: mean,
          maxx: width - mean,
          maxy: height - mean,
          khalf: mean,
        }
      });
    kernel.setOutput([width, height]);
    kernel.floatTextures = true;
    return kernel;
  } 
  convolution(input, output, kernel) {
    var out = this.blurGPUKernel(GPU.input(input.pixel, [input.x, input.y]), GPU.input(kernel.matrix, [kernel.size, kernel.size]));
    for (let x = 0; x < output.x; x++) {
      for (let y = 0; y < output.y; y++) {
        output.pixel[x + output.x * y] = out[y][x];
      }
    }
  }
  computeFrame() {
    this.canvasContext1.drawImage(this.video, 0, 0, this.state.width, this.state.height);
    let frame = this.canvasContext1.getImageData(0, 0, this.state.width, this.state.height);
    let l = frame.data.length / 4;
    let image = {
      x: this.state.width,
      y: this.state.height,
      pixel: this.mainImage,
      max: 16581375
    };
    let i;
    for (i = 0; i < l; i++) {
      let r = frame.data[i * 4 + 0];
      let g = frame.data[i * 4 + 1];
      let b = frame.data[i * 4 + 2];
      image.pixel[i] = ((r << 16) + (g << 8) + b);
    }
    this.doubleBuffer = image.pixel;
    let buf = {
      x: image.x,
      y: image.y,
      pixel: this.doubleBuffer,
      max: image.max
    };
    this.convolution(buf, image, this.blurMatrix);
    
    for (i = 0; i < l; i++) {
      let fpixel = image.pixel[i] / (image.max / 255);
      frame.data[i * 4] = fpixel;
      frame.data[i * 4 + 1] = fpixel;
      frame.data[i * 4 + 2] = fpixel;
      /*frame.data[i * 4 + 0] = fpixel >> 16 & 0xFF;
      frame.data[i * 4 + 1] = fpixel >> 8 & 0xFF;
      frame.data[i * 4 + 2] = fpixel & 0xFF;*/
    }
    this.canvasContext2.putImageData(frame, 0, 0);
  }

  render() {
    return (
      <div className={styles.section}>
        <h1>html5 Canvas RealTime GPU filter</h1>
        <video id="video" src="small.mp4" controls="true" ref={(n) => {this.video = n}}></video>
        <div>
          <canvas id="c1" style={{display: 'none'}} ref={(n) => {this.canvas1 = n}}></canvas>
          <canvas id="c2" ref={(n) => {this.canvas2 = n}}></canvas>
        </div>
      </div>
    );
  }
}



Main.propTypes = {
};

export default Main;
